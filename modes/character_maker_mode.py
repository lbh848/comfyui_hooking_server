"""
Character Maker

캐릭터 확정 전까지의 세계관, 대화, 자유 편집 태그, 참고 이미지와 생성 결과를
서버 프로세스 수명에만 묶어 관리한다. 이 모듈은 config.json 또는 tags.json을
자동으로 수정하지 않는다. tags.json 반영은 confirm()의 명시적 확정 단계에서만
백업 후 원자적으로 수행한다.
"""

from __future__ import annotations

import asyncio
import base64
import copy
import datetime
import importlib
import io
import json
import math
import os
import shutil
import tempfile
import time
import traceback
import uuid
from typing import Any, Callable

from PIL import Image, ImageDraw

from . import llm_prompt_edit
from . import llm_service
from .danbooru_rag import DanbooruRagError, get_danbooru_rag_service


EDITABLE_FIELDS = ("appearance", "outfit", "expression", "composition")
RAG_COLD_START_TIMEOUT_SECONDS = 300.0
MAX_REFERENCE_COUNT = 8
MAX_REFERENCE_BYTES = 12 * 1024 * 1024
MAX_CHAT_ITEMS = 40
MAX_REVISIONS = 12


class CharacterMakerError(ValueError):
    """사용자 입력 또는 현재 세션 상태로 작업을 진행할 수 없을 때 사용."""


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _normalize_tag_list(value: Any, *, field: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise CharacterMakerError(f"{field} 태그는 문자열 배열이어야 합니다.")
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in value:
        if not isinstance(raw, str):
            raise CharacterMakerError(f"{field} 태그에는 문자열만 사용할 수 있습니다.")
        tag = raw.strip()
        if not tag:
            continue
        if len(tag) > 300:
            raise CharacterMakerError(f"{field} 태그 한 항목은 300자를 넘을 수 없습니다.")
        key = tag.casefold()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(tag)
    if len(normalized) > 160:
        raise CharacterMakerError(f"{field} 태그는 160개를 넘을 수 없습니다.")
    return normalized


def _normalize_fields(value: Any) -> dict[str, list[str]]:
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise CharacterMakerError("fields는 객체여야 합니다.")
    return {
        field: _normalize_tag_list(value.get(field, []), field=field)
        for field in EDITABLE_FIELDS
    }


def _normalize_locks(value: Any) -> dict[str, bool]:
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise CharacterMakerError("locks는 객체여야 합니다.")
    return {field: bool(value.get(field, False)) for field in EDITABLE_FIELDS}


def _normalize_lora_list(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise CharacterMakerError("캐릭터 메이커 LoRA 목록은 배열이어야 합니다.")
    if len(value) > 12:
        raise CharacterMakerError("캐릭터 메이커 LoRA는 최대 12개까지 사용할 수 있습니다.")

    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for index, raw in enumerate(value):
        if not isinstance(raw, dict):
            raise CharacterMakerError(f"LoRA {index + 1}번 항목은 객체여야 합니다.")
        source = str(raw.get("source") or "asset").strip().lower()
        if source not in {"asset", "bot", "instance"}:
            raise CharacterMakerError(
                f"LoRA {index + 1}번 항목의 source가 올바르지 않습니다."
            )
        lora_path = str(raw.get("lora_path") or "").strip()
        if not lora_path or len(lora_path) > 1000:
            raise CharacterMakerError(
                f"LoRA {index + 1}번 항목의 모델 경로가 비어 있거나 너무 깁니다."
            )
        path_parts = [
            part for part in lora_path.replace("\\", "/").split("/") if part
        ]
        if os.path.isabs(lora_path) or ".." in path_parts:
            raise CharacterMakerError(
                f"LoRA {index + 1}번 항목은 상대 모델 경로만 사용할 수 있습니다."
            )
        base = str(raw.get("BASE") or "anima").strip().lower()
        if base == "ilxl":
            base = "sdxl"
        if base not in {"anima", "sdxl"}:
            raise CharacterMakerError(
                f"LoRA {index + 1}번 항목의 BASE는 anima 또는 sdxl이어야 합니다."
            )
        try:
            strength = float(raw.get("strength", 0.5))
        except (TypeError, ValueError) as exc:
            raise CharacterMakerError(
                f"LoRA {index + 1}번 항목의 강도가 숫자가 아닙니다."
            ) from exc
        if not math.isfinite(strength) or not 0.0 <= strength <= 2.0:
            raise CharacterMakerError(
                f"LoRA {index + 1}번 항목의 강도는 0~2 사이여야 합니다."
            )
        identity = (source, lora_path.casefold(), base)
        if identity in seen:
            continue
        seen.add(identity)
        normalized.append(
            {
                "name": str(raw.get("name") or os.path.basename(lora_path)).strip()[:300],
                "character": str(raw.get("character") or "").strip()[:300],
                "lora_path": lora_path,
                "strength": strength,
                "preview_url": str(raw.get("preview_url") or "").strip()[:2000],
                "trigger": str(raw.get("trigger") or "").strip()[:1000],
                "BASE": base,
                "source": source,
                "lora_id": str(raw.get("lora_id") or "").strip()[:300],
            }
        )
    return normalized


def _parse_llm_payload(raw: str, *, require_queries: bool) -> dict[str, Any] | None:
    parsed = llm_prompt_edit.parse_llm_json(raw)
    if not isinstance(parsed, dict):
        return None
    assistant_message = parsed.get("assistant_message")
    fields = parsed.get("fields")
    if not isinstance(assistant_message, str) or not assistant_message.strip():
        return None
    if not isinstance(fields, dict):
        return None
    if set(fields) != set(EDITABLE_FIELDS):
        return None
    try:
        normalized_fields = _normalize_fields(fields)
    except CharacterMakerError:
        return None

    raw_queries = parsed.get("rag_queries", {})
    if require_queries and not isinstance(raw_queries, dict):
        return None
    if require_queries and set(raw_queries) != set(EDITABLE_FIELDS):
        return None
    if not isinstance(raw_queries, dict):
        raw_queries = {}
    rag_queries: dict[str, list[str]] = {}
    for field in EDITABLE_FIELDS:
        queries = raw_queries.get(field, [])
        if not isinstance(queries, list):
            return None
        clean_queries: list[str] = []
        for query in queries:
            if not isinstance(query, str):
                return None
            query = query.strip()
            if query and query not in clean_queries:
                clean_queries.append(query[:300])
        rag_queries[field] = clean_queries[:4]

    return {
        "assistant_message": assistant_message.strip()[:4000],
        "fields": normalized_fields,
        "rag_queries": rag_queries,
    }


def validate_character_maker_llm_result(
    raw: str, *, require_queries: bool = True
) -> tuple[bool, str]:
    """callLLMTask/callLLMVisionTask result_validator 계약."""
    parsed = _parse_llm_payload(raw, require_queries=require_queries)
    if parsed is None:
        return (
            False,
            "assistant_message, fields(4개 배열), rag_queries(4개 배열)를 가진 JSON 객체가 필요합니다.",
        )
    return True, ""


def _tag_diff(
    before: dict[str, list[str]], after: dict[str, list[str]]
) -> dict[str, dict[str, list[str]]]:
    result: dict[str, dict[str, list[str]]] = {}
    for field in EDITABLE_FIELDS:
        old = before.get(field, [])
        new = after.get(field, [])
        old_keys = {item.casefold() for item in old}
        new_keys = {item.casefold() for item in new}
        result[field] = {
            "added": [item for item in new if item.casefold() not in old_keys],
            "removed": [item for item in old if item.casefold() not in new_keys],
        }
    return result


def _assert_within(root: str, target: str) -> None:
    root_real = os.path.realpath(root)
    target_real = os.path.realpath(target)
    try:
        common = os.path.commonpath([root_real, target_real])
    except ValueError as exc:
        raise CharacterMakerError(f"임시 경로 검증 실패: {target_real}") from exc
    if common != root_real or target_real == root_real:
        raise CharacterMakerError(f"임시 경로가 작업 루트를 벗어났습니다: {target_real}")


def _safe_registration_name(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise CharacterMakerError(f"{label}은 문자열이어야 합니다.")
    name = value.strip()
    if not name:
        raise CharacterMakerError(f"{label}을 입력하세요.")
    if len(name) > 100:
        raise CharacterMakerError(f"{label}은 100자를 넘을 수 없습니다.")
    if any(ord(ch) < 32 for ch in name) or any(ch in name for ch in '<>:"/\\|?*'):
        raise CharacterMakerError(f"{label}에 파일명 금지 문자를 사용할 수 없습니다.")
    return name


class CharacterMakerService:
    def __init__(
        self,
        asset_manager: Any,
        config_getter: Callable[[], dict[str, Any]],
        *,
        temp_root: str,
    ) -> None:
        self.asset_manager = asset_manager
        self.config_getter = config_getter
        self.temp_root = os.path.realpath(temp_root)
        os.makedirs(self.temp_root, exist_ok=True)
        self.boot_id = uuid.uuid4().hex
        self.sessions: dict[str, dict[str, Any]] = {}
        self._operation_locks: dict[str, asyncio.Lock] = {}

    def _default_settings(self) -> dict[str, Any]:
        config = self.config_getter() or {}
        return {
            "asset_workflow_type": str(config.get("asset_workflow_type") or "ilxl"),
            "quality_preset": "",
            "artist_preset": "",
            "negative_preset": "",
            "character_negative_preset": "",
            "natural_language_preset": "",
            "anima_quality_preset": "",
            "anima_artist_preset": "",
            "anima_negative_preset": "",
            "img_w": 700,
            "img_h": 1024,
            "seed": -1,
            "rag_enabled": bool(config.get("character_maker_rag_enabled", False)),
            "lora_enabled": False,
            "lora_list": [],
            # 생성 옵션 — 에셋/오토매치와 동일 프롬프트 토큰 세트.
            # pose·ipadapter(참조 이미지)는 CM 미지원 잠금으로 유지한다.
            "style_lora_enabled": False,
            "style_lora_list": [],
            "face_lora_enabled": False,
            "face_lora_list": [],
            "face_lora_upscale_size": "",
            "face_tags": "",
            "eye_tags": "",
            "hrf_sdxl": False,
            "hrf_anima": False,
            "hrf_size": 2.0,
            "hrf_restore_size": True,
            "hrf_control_net": False,
            "sdxl_fd_enabled": False,
            "sdxl_hd_enabled": False,
            "sdxl_ed_enabled": False,
            "anima_fd_enabled": False,
            "anima_hd_enabled": False,
            "anima_ed_enabled": False,
            "face_crop_top": 2.5,
            "face_crop_bottom": 1.0,
        }

    def create_session(self) -> dict[str, Any]:
        session_id = uuid.uuid4().hex
        session_dir = os.path.join(self.temp_root, session_id)
        os.makedirs(os.path.join(session_dir, "references"), exist_ok=True)
        os.makedirs(os.path.join(session_dir, "images"), exist_ok=True)
        now = _now_iso()
        session = {
            "id": session_id,
            "boot_id": self.boot_id,
            "created_at": now,
            "updated_at": now,
            "world_context": "",
            "fields": {field: [] for field in EDITABLE_FIELDS},
            "locks": {field: False for field in EDITABLE_FIELDS},
            "settings": self._default_settings(),
            "chat": [],
            "references": [],
            "revisions": [],
            "active_revision_id": "",
            "finalized": None,
        }
        self.sessions[session_id] = session
        self._operation_locks[session_id] = asyncio.Lock()
        print(f"[CHARACTER_MAKER] 임시 세션 생성: session={session_id}, boot={self.boot_id}")
        return self.public_session(session_id)

    def _session(self, session_id: str) -> dict[str, Any]:
        session = self.sessions.get(str(session_id or "").strip())
        if session is None:
            print(
                f"[CHARACTER_MAKER] 세션 조회 실패: session={session_id!r}, "
                f"boot={self.boot_id}, active={len(self.sessions)}"
            )
            raise CharacterMakerError(
                "임시 세션이 없거나 서버가 재시작되었습니다. 새 세션을 시작하세요."
            )
        return session

    def public_session(self, session_id: str) -> dict[str, Any]:
        session = self._session(session_id)
        active_revision_id = session.get("active_revision_id", "")
        references = [
            {
                "id": item["id"],
                "name": item["name"],
                "mime": item["mime"],
                "url": f"/api/character_maker/session/{session_id}/reference/{item['id']}",
            }
            for item in session["references"]
        ]
        revisions = [
            {
                "id": item["id"],
                "created_at": item["created_at"],
                "fields": copy.deepcopy(item["fields"]),
                "note": item.get("note", ""),
                "url": f"/api/character_maker/session/{session_id}/image/{item['id']}",
                "active": item["id"] == active_revision_id,
            }
            for item in session["revisions"]
        ]
        return {
            "id": session["id"],
            "boot_id": session["boot_id"],
            "created_at": session["created_at"],
            "updated_at": session["updated_at"],
            "world_context": session["world_context"],
            "fields": copy.deepcopy(session["fields"]),
            "locks": copy.deepcopy(session["locks"]),
            "settings": copy.deepcopy(session["settings"]),
            "chat": copy.deepcopy(session["chat"]),
            "references": references,
            "revisions": revisions,
            "active_revision_id": active_revision_id,
            "finalized": copy.deepcopy(session.get("finalized")),
        }

    def operation_lock(self, session_id: str) -> asyncio.Lock:
        session = self._session(session_id)
        lock = self._operation_locks.get(session["id"])
        if lock is None:
            print(
                f"[CHARACTER_MAKER] 작업 잠금 캐시 미스, 재생성: "
                f"session={session['id']}"
            )
            lock = asyncio.Lock()
            self._operation_locks[session["id"]] = lock
        return lock

    def update_session(self, session_id: str, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise CharacterMakerError("세션 변경값은 객체여야 합니다.")
        session = self._session(session_id)
        if "world_context" in payload:
            world_context = payload.get("world_context")
            if not isinstance(world_context, str):
                raise CharacterMakerError("세계관 정보는 문자열이어야 합니다.")
            session["world_context"] = world_context[:20000]
        if "fields" in payload:
            session["fields"] = _normalize_fields(payload.get("fields"))
        if "locks" in payload:
            session["locks"] = _normalize_locks(payload.get("locks"))
        if "settings" in payload:
            self._update_settings(session, payload.get("settings"))
        if "active_revision_id" in payload:
            revision_id = str(payload.get("active_revision_id") or "")
            if revision_id and not any(
                item["id"] == revision_id for item in session["revisions"]
            ):
                raise CharacterMakerError("선택한 리비전을 찾을 수 없습니다.")
            session["active_revision_id"] = revision_id
        session["updated_at"] = _now_iso()
        return self.public_session(session_id)

    def _update_settings(self, session: dict[str, Any], raw: Any) -> None:
        if not isinstance(raw, dict):
            raise CharacterMakerError("제작 설정은 객체여야 합니다.")
        settings = session["settings"]
        string_fields = (
            "asset_workflow_type",
            "quality_preset",
            "artist_preset",
            "negative_preset",
            "character_negative_preset",
            "natural_language_preset",
            "anima_quality_preset",
            "anima_artist_preset",
            "anima_negative_preset",
        )
        for key in string_fields:
            if key in raw:
                value = raw.get(key)
                if not isinstance(value, str):
                    raise CharacterMakerError(f"{key} 설정은 문자열이어야 합니다.")
                settings[key] = value.strip()[:200]
        for key, low, high in (("img_w", 256, 4096), ("img_h", 256, 4096)):
            if key in raw:
                value = int(raw.get(key))
                if not low <= value <= high:
                    raise CharacterMakerError(f"{key}는 {low}~{high} 사이여야 합니다.")
                settings[key] = value
        if "seed" in raw:
            seed = int(raw.get("seed"))
            if not -1 <= seed <= 2**32 - 1:
                raise CharacterMakerError("seed는 -1 또는 0~4294967295 사이여야 합니다.")
            settings["seed"] = seed
        for key in ("rag_enabled", "lora_enabled"):
            if key in raw:
                settings[key] = bool(raw.get(key))
        for key in (
            "style_lora_enabled",
            "face_lora_enabled",
            "hrf_sdxl",
            "hrf_anima",
            "hrf_restore_size",
            "hrf_control_net",
            "sdxl_fd_enabled",
            "sdxl_hd_enabled",
            "sdxl_ed_enabled",
            "anima_fd_enabled",
            "anima_hd_enabled",
            "anima_ed_enabled",
        ):
            if key in raw:
                settings[key] = bool(raw.get(key))
        if "lora_list" in raw:
            settings["lora_list"] = _normalize_lora_list(raw.get("lora_list"))
        for key in ("style_lora_list", "face_lora_list"):
            if key in raw:
                settings[key] = _normalize_lora_list(raw.get(key))
        for key in ("face_tags", "eye_tags", "face_lora_upscale_size"):
            if key in raw:
                value = raw.get(key)
                if not isinstance(value, (str, int, float)) or isinstance(value, bool):
                    raise CharacterMakerError(f"{key} 설정은 문자열 또는 숫자여야 합니다.")
                settings[key] = str(value).strip()[:500]
        if "hrf_size" in raw:
            try:
                hrf_size = float(raw.get("hrf_size"))
            except (TypeError, ValueError) as exc:
                raise CharacterMakerError("hrf_size는 숫자여야 합니다.") from exc
            if not math.isfinite(hrf_size) or not 1.0 <= hrf_size <= 3.0:
                raise CharacterMakerError("hrf_size는 1.0~3.0 사이여야 합니다.")
            settings["hrf_size"] = hrf_size
        for key in ("face_crop_top", "face_crop_bottom"):
            if key in raw:
                try:
                    value = float(raw.get(key))
                except (TypeError, ValueError) as exc:
                    raise CharacterMakerError(f"{key}는 숫자여야 합니다.") from exc
                if not math.isfinite(value) or not 0.0 <= value <= 10.0:
                    raise CharacterMakerError(f"{key}는 0.0~10.0 사이여야 합니다.")
                settings[key] = value

    def add_reference(
        self, session_id: str, *, filename: str, mime: str, image_bytes: bytes
    ) -> dict[str, Any]:
        session = self._session(session_id)
        if len(session["references"]) >= MAX_REFERENCE_COUNT:
            raise CharacterMakerError(f"참고 이미지는 최대 {MAX_REFERENCE_COUNT}장입니다.")
        if not image_bytes:
            raise CharacterMakerError("참고 이미지 데이터가 비어 있습니다.")
        if len(image_bytes) > MAX_REFERENCE_BYTES:
            raise CharacterMakerError("참고 이미지 한 장은 12MB를 넘을 수 없습니다.")
        try:
            with Image.open(io.BytesIO(image_bytes)) as image:
                image.verify()
            with Image.open(io.BytesIO(image_bytes)) as image:
                fmt = (image.format or "").upper()
        except Exception as exc:
            print(
                f"[CHARACTER_MAKER] 참고 이미지 검증 실패: "
                f"session={session_id}, filename={filename!r}, error={exc}"
            )
            traceback.print_exc()
            raise CharacterMakerError("올바른 이미지 파일이 아닙니다.") from exc

        ext_map = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "GIF": ".gif"}
        ext = ext_map.get(fmt, ".img")
        detected_mime = {
            "JPEG": "image/jpeg",
            "PNG": "image/png",
            "WEBP": "image/webp",
            "GIF": "image/gif",
        }.get(fmt, mime or "application/octet-stream")
        reference_id = uuid.uuid4().hex
        path = os.path.join(self.temp_root, session_id, "references", reference_id + ext)
        _assert_within(self.temp_root, path)
        with open(path, "wb") as handle:
            handle.write(image_bytes)
        item = {
            "id": reference_id,
            "name": os.path.basename(filename or f"reference{ext}")[:200],
            "mime": detected_mime,
            "path": path,
        }
        session["references"].append(item)
        session["updated_at"] = _now_iso()
        print(
            f"[CHARACTER_MAKER] 참고 이미지 추가: session={session_id}, "
            f"reference={reference_id}, bytes={len(image_bytes)}, format={fmt}"
        )
        return self.public_session(session_id)

    def remove_reference(self, session_id: str, reference_id: str) -> dict[str, Any]:
        session = self._session(session_id)
        target = next(
            (item for item in session["references"] if item["id"] == reference_id),
            None,
        )
        if target is None:
            print(
                f"[CHARACTER_MAKER] 참고 이미지 삭제 실패: "
                f"session={session_id}, reference={reference_id}"
            )
            raise CharacterMakerError("참고 이미지를 찾을 수 없습니다.")
        path = target["path"]
        _assert_within(self.temp_root, path)
        if os.path.isfile(path):
            os.remove(path)
        session["references"] = [
            item for item in session["references"] if item["id"] != reference_id
        ]
        session["updated_at"] = _now_iso()
        print(
            f"[CHARACTER_MAKER] 참고 이미지 삭제: "
            f"session={session_id}, reference={reference_id}"
        )
        return self.public_session(session_id)

    def reference_path(self, session_id: str, reference_id: str) -> tuple[str, str]:
        session = self._session(session_id)
        item = next(
            (item for item in session["references"] if item["id"] == reference_id),
            None,
        )
        if item is None or not os.path.isfile(item["path"]):
            print(
                f"[CHARACTER_MAKER] 참고 이미지 조회 실패: "
                f"session={session_id}, reference={reference_id}"
            )
            raise CharacterMakerError("참고 이미지를 찾을 수 없습니다.")
        _assert_within(self.temp_root, item["path"])
        return item["path"], item["mime"]

    def _active_revision(self, session: dict[str, Any]) -> dict[str, Any] | None:
        active_id = session.get("active_revision_id", "")
        if not active_id:
            return None
        return next(
            (item for item in session["revisions"] if item["id"] == active_id),
            None,
        )

    def _vision_sheet(self, session: dict[str, Any]) -> tuple[str, str] | None:
        sources: list[tuple[str, str]] = []
        active = self._active_revision(session)
        if active and os.path.isfile(active["image_path"]):
            sources.append(("CURRENT", active["image_path"]))
        for index, item in enumerate(
            session["references"][:MAX_REFERENCE_COUNT], start=1
        ):
            if os.path.isfile(item["path"]):
                sources.append((f"REF {index}", item["path"]))
        if not sources:
            print(
                f"[CHARACTER_MAKER] 비전 입력 없음: "
                f"session={session['id']}, active_revision 없음, reference 없음"
            )
            return None
        try:
            tile_w, tile_h, label_h = 384, 384, 28
            columns = 2 if len(sources) > 1 else 1
            rows = (len(sources) + columns - 1) // columns
            sheet = Image.new("RGB", (tile_w * columns, (tile_h + label_h) * rows), "#121722")
            draw = ImageDraw.Draw(sheet)
            for index, (label, path) in enumerate(sources):
                with Image.open(path) as source:
                    image = source.convert("RGB")
                    image.thumbnail((tile_w, tile_h), Image.Resampling.LANCZOS)
                    x0 = (index % columns) * tile_w
                    y0 = (index // columns) * (tile_h + label_h)
                    x = x0 + (tile_w - image.width) // 2
                    y = y0 + label_h + (tile_h - image.height) // 2
                    sheet.paste(image, (x, y))
                    draw.rectangle((x0, y0, x0 + tile_w, y0 + label_h), fill="#202a3b")
                    draw.text((x0 + 10, y0 + 8), label, fill="#f4f7fb")
            output = io.BytesIO()
            sheet.save(output, format="WEBP", quality=88, method=4)
            return base64.b64encode(output.getvalue()).decode("ascii"), "image/webp"
        except Exception as exc:
            print(
                f"[CHARACTER_MAKER] 비전 입력 시트 생성 실패: "
                f"session={session['id']}, sources={len(sources)}, error={exc}"
            )
            traceback.print_exc()
            raise CharacterMakerError("비전 입력 이미지를 준비하지 못했습니다.") from exc

    def _revision_messages(
        self, session: dict[str, Any], feedback: str, *, rag_enabled: bool
    ) -> list[dict[str, str]]:
        system = (
            "You are a collaborative character-design editor for an image-generation UI. "
            "Reason from the complete world context, conversation, current visual evidence, "
            "references, and user feedback. Do not use hard-coded keyword matching. "
            "You may modify only appearance, outfit, expression, and composition. "
            "All other generation settings are read-only presets. Preserve locked fields exactly. "
            "Return one JSON object with assistant_message, fields, and rag_queries. "
            "fields must contain exactly appearance/outfit/expression/composition string arrays. "
            "rag_queries must contain the same four keys with short Korean or English semantic "
            "search units. Tags should describe visible, image-generatable details. "
            f"Danbooru RAG is {'enabled' if rag_enabled else 'disabled'}."
        )
        history = session["chat"][:-1][-12:]
        user_payload = {
            "world_context": session["world_context"],
            "feedback": feedback,
            "current_fields": session["fields"],
            "locks": session["locks"],
            "read_only_settings": session["settings"],
            "recent_conversation": history,
            "image_legend": "CURRENT is the latest generated image; REF N are user references.",
        }
        return [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": "캐릭터 상태를 수정하고 JSON만 반환하세요.\n"
                + json.dumps(user_payload, ensure_ascii=False, indent=2),
            },
        ]

    async def _call_revision_llm(
        self,
        task_key: str,
        messages: list[dict[str, str]],
        vision: tuple[str, str] | None,
        *,
        require_queries: bool,
    ) -> dict[str, Any]:
        validator = lambda raw: validate_character_maker_llm_result(
            raw, require_queries=require_queries
        )
        if vision is not None:
            raw = await llm_service.callLLMVisionTask(
                task_key,
                messages,
                image_b64=vision[0],
                image_mime=vision[1],
                json_mode=True,
                result_validator=validator,
            )
        else:
            raw = await llm_service.callLLMTask(
                task_key,
                messages,
                json_mode=True,
                result_validator=validator,
            )
        if isinstance(raw, str) and raw.strip().startswith("[LLM 실패]"):
            print(f"[CHARACTER_MAKER] LLM 호출 최종 실패: task={task_key}, result={raw}")
            raise CharacterMakerError(raw)
        parsed = _parse_llm_payload(raw, require_queries=require_queries)
        if parsed is None:
            print(
                f"[CHARACTER_MAKER] LLM JSON 검증 실패: task={task_key}, "
                f"raw={str(raw)[:1000]!r}"
            )
            raise CharacterMakerError("LLM 응답 형식이 올바르지 않습니다.")
        return parsed

    async def _ensure_rag_ready(self, service: Any) -> dict[str, Any]:
        """Allow model/index cold start without weakening search timeouts."""
        status = await asyncio.to_thread(service.status)
        if status.get("loaded"):
            return {
                "success": True,
                "loaded": True,
                "row_count": int(status.get("row_count") or 0),
                "variant": str(status.get("variant") or "b"),
                "mode": "embedded",
            }
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(service.warmup),
                timeout=RAG_COLD_START_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError as exc:
            print(
                "[CHARACTER_MAKER_RAG] 내장 서비스 최초 준비 시간 초과: "
                f"timeout={RAG_COLD_START_TIMEOUT_SECONDS}"
            )
            traceback.print_exc()
            raise CharacterMakerError(
                "내장 RAG 최초 준비가 "
                f"{RAG_COLD_START_TIMEOUT_SECONDS:g}초 안에 끝나지 않았습니다."
            ) from exc

    async def _rag_search(
        self, query: str, *, config: dict[str, Any]
    ) -> list[dict[str, Any]]:
        top_k = max(1, min(20, int(config.get("character_maker_rag_top_k", 5))))
        threshold = float(config.get("character_maker_rag_threshold", 0.0))
        timeout_seconds = max(
            2.0, min(120.0, float(config.get("character_maker_rag_timeout_sec", 20.0)))
        )
        service = get_danbooru_rag_service()
        await self._ensure_rag_ready(service)
        try:
            results = await asyncio.wait_for(
                asyncio.to_thread(
                    service.search,
                    query,
                    top_k=top_k,
                    threshold=threshold,
                    categories={0},
                ),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            print(
                "[CHARACTER_MAKER_RAG] 내장 검색 시간 초과: "
                f"query={query!r}, timeout={timeout_seconds}"
            )
            traceback.print_exc()
            raise CharacterMakerError(
                f"내장 RAG 검색이 {timeout_seconds:g}초 안에 끝나지 않았습니다."
            ) from exc
        except DanbooruRagError as exc:
            print(
                "[CHARACTER_MAKER_RAG] 내장 검색 실패: "
                f"query={query!r}, error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            raise CharacterMakerError(str(exc)) from exc
        except Exception as exc:
            print(
                "[CHARACTER_MAKER_RAG] 내장 검색 예외: "
                f"query={query!r}, error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            raise CharacterMakerError(f"내장 RAG 검색 실패: {exc}") from exc
        if not isinstance(results, list):
            print(
                f"[CHARACTER_MAKER_RAG] 검색 결과 누락: query={query!r}, "
                f"result_type={type(results).__name__}"
            )
            raise CharacterMakerError("RAG 검색 결과 배열이 없습니다.")
        return [item for item in results if isinstance(item, dict) and item.get("tag")]

    async def _rag_refine(
        self,
        *,
        task_key: str,
        session: dict[str, Any],
        draft: dict[str, Any],
        config: dict[str, Any],
    ) -> tuple[dict[str, list[str]], dict[str, Any]]:
        jobs: list[tuple[str, str]] = []
        for field in EDITABLE_FIELDS:
            if session["locks"][field]:
                continue
            queries = list(draft["rag_queries"].get(field, []))
            if not queries:
                queries = list(draft["fields"].get(field, []))
            for query in queries[:4]:
                jobs.append((field, query))
        if not jobs:
            print(
                f"[CHARACTER_MAKER_RAG] 검색 단위 없음: session={session['id']}, "
                "현재 필드를 유지합니다."
            )
            return copy.deepcopy(session["fields"]), {"queries": [], "candidates": {}}

        unique_queries = list(dict.fromkeys(query for _, query in jobs))
        semaphore = asyncio.Semaphore(4)

        async def _bounded_search(query: str) -> tuple[str, list[dict[str, Any]]]:
            async with semaphore:
                return query, await self._rag_search(query, config=config)

        unique_results = await asyncio.gather(
            *(_bounded_search(query) for query in unique_queries)
        )
        hits_by_query = dict(unique_results)
        candidates: dict[str, list[dict[str, Any]]] = {
            field: [] for field in EDITABLE_FIELDS
        }
        for field in EDITABLE_FIELDS:
            for tag in session["fields"][field]:
                candidates[field].append(
                    {
                        "query": "(current user value)",
                        "tag": tag,
                        "score": None,
                        "definition": "User-authored current tag; trusted for preservation.",
                        "aliases": [],
                    }
                )
        for field, query in jobs:
            hits = hits_by_query.get(query, [])
            for hit in hits:
                candidate = {
                    "query": query,
                    "tag": str(hit.get("tag") or ""),
                    "score": hit.get("score"),
                    "definition": str(hit.get("definition") or "")[:500],
                    "aliases": hit.get("aliases"),
                }
                if candidate["tag"] and not any(
                    item["tag"].casefold() == candidate["tag"].casefold()
                    for item in candidates[field]
                ):
                    candidates[field].append(candidate)

        selection_messages = [
            {
                "role": "system",
                "content": (
                    "Select Danbooru tags for the character edit. Do not use hard-coded keyword "
                    "matching. For every unlocked field, fields may contain only exact tag strings "
                    "present in that field's candidate_pool. Preserve locked fields exactly. "
                    "Return JSON with assistant_message, fields, and rag_queries (empty arrays allowed)."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "world_context": session["world_context"],
                        "current_fields": session["fields"],
                        "locks": session["locks"],
                        "draft_intent": draft,
                        "candidate_pool": candidates,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            },
        ]
        selected = await self._call_revision_llm(
            task_key, selection_messages, None, require_queries=False
        )

        final_fields: dict[str, list[str]] = {}
        dropped: dict[str, list[str]] = {}
        for field in EDITABLE_FIELDS:
            if session["locks"][field]:
                final_fields[field] = list(session["fields"][field])
                dropped[field] = []
                continue
            allowed = {
                str(item.get("tag") or "").casefold(): str(item.get("tag") or "")
                for item in candidates[field]
                if str(item.get("tag") or "").strip()
            }
            if not allowed:
                final_fields[field] = list(session["fields"][field])
                dropped[field] = list(selected["fields"][field])
                print(
                    f"[CHARACTER_MAKER_RAG] 후보 없음으로 현재값 유지: "
                    f"session={session['id']}, field={field}"
                )
                continue
            accepted: list[str] = []
            rejected: list[str] = []
            for tag in selected["fields"][field]:
                canonical = allowed.get(tag.casefold())
                if canonical and canonical not in accepted:
                    accepted.append(canonical)
                else:
                    rejected.append(tag)
            final_fields[field] = accepted
            dropped[field] = rejected
            if rejected:
                print(
                    f"[CHARACTER_MAKER_RAG] 후보 외 태그 제거: "
                    f"session={session['id']}, field={field}, tags={rejected}"
                )
        return final_fields, {
            "queries": [{"field": field, "query": query} for field, query in jobs],
            "candidates": candidates,
            "dropped": dropped,
        }

    async def revise(self, session_id: str, payload: Any) -> dict[str, Any]:
        async with self.operation_lock(session_id):
            return await self._revise_locked(session_id, payload)

    async def _revise_locked(self, session_id: str, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise CharacterMakerError("수정 요청은 객체여야 합니다.")
        session = self._session(session_id)
        self.update_session(
            session_id,
            {
                key: payload[key]
                for key in ("world_context", "fields", "locks", "settings")
                if key in payload
            },
        )
        feedback = payload.get("message")
        if not isinstance(feedback, str) or not feedback.strip():
            raise CharacterMakerError("LLM에게 전달할 요청이나 피드백을 입력하세요.")
        feedback = feedback.strip()[:8000]
        before = copy.deepcopy(session["fields"])
        session["chat"].append(
            {"id": uuid.uuid4().hex, "role": "user", "content": feedback, "at": _now_iso()}
        )
        session["chat"] = session["chat"][-MAX_CHAT_ITEMS:]

        config = self.config_getter() or {}
        rag_enabled = bool(session["settings"].get("rag_enabled", False))
        active = self._active_revision(session)
        task_key = "character_maker_feedback" if active else "character_maker_draft"
        vision = self._vision_sheet(session)
        messages = self._revision_messages(session, feedback, rag_enabled=rag_enabled)
        draft = await self._call_revision_llm(
            task_key, messages, vision, require_queries=True
        )

        for field in EDITABLE_FIELDS:
            if session["locks"][field]:
                draft["fields"][field] = list(before[field])

        rag_meta: dict[str, Any] = {"enabled": False}
        if rag_enabled:
            refined_fields, rag_details = await self._rag_refine(
                task_key=task_key,
                session=session,
                draft=draft,
                config=config,
            )
            final_fields = refined_fields
            rag_meta = {"enabled": True, **rag_details}
        else:
            final_fields = draft["fields"]

        for field in EDITABLE_FIELDS:
            if session["locks"][field]:
                final_fields[field] = list(before[field])
        session["fields"] = _normalize_fields(final_fields)
        assistant_message = draft["assistant_message"]
        session["chat"].append(
            {
                "id": uuid.uuid4().hex,
                "role": "assistant",
                "content": assistant_message,
                "at": _now_iso(),
            }
        )
        session["chat"] = session["chat"][-MAX_CHAT_ITEMS:]
        session["updated_at"] = _now_iso()
        print(
            f"[CHARACTER_MAKER] LLM 수정 완료: session={session_id}, "
            f"task={task_key}, vision={vision is not None}, rag={rag_enabled}"
        )
        return {
            "success": True,
            "session": self.public_session(session_id),
            "diff": _tag_diff(before, session["fields"]),
            "rag": rag_meta,
        }

    def add_revision(
        self,
        session_id: str,
        *,
        image_path: str,
        prompt_path: str,
        positive: str,
        negative: str,
        note: str = "",
    ) -> dict[str, Any]:
        session = self._session(session_id)
        _assert_within(self.temp_root, image_path)
        _assert_within(self.temp_root, prompt_path)
        if not os.path.isfile(image_path):
            raise CharacterMakerError("생성된 임시 이미지 파일을 찾을 수 없습니다.")
        revision_id = uuid.uuid4().hex
        item = {
            "id": revision_id,
            "created_at": _now_iso(),
            "fields": copy.deepcopy(session["fields"]),
            "settings": copy.deepcopy(session["settings"]),
            "image_path": image_path,
            "prompt_path": prompt_path,
            "positive": positive,
            "negative": negative,
            "note": str(note or "")[:1000],
        }
        session["revisions"].append(item)
        session["active_revision_id"] = revision_id
        while len(session["revisions"]) > MAX_REVISIONS:
            removed = session["revisions"].pop(0)
            for key in ("image_path", "prompt_path"):
                path = removed.get(key)
                if path and os.path.isfile(path):
                    try:
                        _assert_within(self.temp_root, path)
                        os.remove(path)
                    except Exception as exc:
                        print(
                            f"[CHARACTER_MAKER] 오래된 리비전 파일 정리 실패: "
                            f"session={session_id}, path={path}, error={exc}"
                        )
                        traceback.print_exc()
        session["updated_at"] = _now_iso()
        print(
            f"[CHARACTER_MAKER] 리비전 추가: session={session_id}, "
            f"revision={revision_id}, count={len(session['revisions'])}"
        )
        return self.public_session(session_id)

    def revision_path(self, session_id: str, revision_id: str) -> tuple[str, str]:
        session = self._session(session_id)
        item = next(
            (item for item in session["revisions"] if item["id"] == revision_id),
            None,
        )
        if item is None or not os.path.isfile(item["image_path"]):
            print(
                f"[CHARACTER_MAKER] 리비전 이미지 조회 실패: "
                f"session={session_id}, revision={revision_id}"
            )
            raise CharacterMakerError("리비전 이미지를 찾을 수 없습니다.")
        _assert_within(self.temp_root, item["image_path"])
        ext = os.path.splitext(item["image_path"])[1].lower()
        mime = {
            ".webp": "image/webp",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
        }.get(ext, "application/octet-stream")
        return item["image_path"], mime

    async def test_rag(
        self, query: str = "", config_override: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        config = copy.deepcopy(self.config_getter() or {})
        if config_override:
            config.update(config_override)
        started = time.perf_counter()
        service = get_danbooru_rag_service()
        try:
            health = await self._ensure_rag_ready(service)
            results = []
            if query.strip():
                results = await self._rag_search(query.strip()[:300], config=config)
            else:
                print("[CHARACTER_MAKER_RAG] 검색어 없이 내장 상태 확인만 완료")
        except CharacterMakerError:
            raise
        except DanbooruRagError as exc:
            print(
                "[CHARACTER_MAKER_RAG] 내장 테스트 실패: "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            raise CharacterMakerError(str(exc)) from exc
        except Exception as exc:
            print(
                "[CHARACTER_MAKER_RAG] 내장 테스트 예외: "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            raise CharacterMakerError(f"내장 RAG 테스트 실패: {exc}") from exc
        return {
            "success": True,
            "health": health,
            "results": results,
            "elapsed_ms": round((time.perf_counter() - started) * 1000),
        }

    @staticmethod
    def _atomic_write_json(path: str, value: dict[str, Any]) -> None:
        directory = os.path.dirname(path)
        os.makedirs(directory, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(prefix=".character_maker_", suffix=".json", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(value, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, path)
        except Exception:
            if os.path.isfile(temp_path):
                os.remove(temp_path)
            raise

    def confirm(self, session_id: str, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise CharacterMakerError("확정 요청은 객체여야 합니다.")
        session = self._session(session_id)
        character_name = _safe_registration_name(payload.get("character_name"), "캐릭터명")
        appearance_name = _safe_registration_name(payload.get("appearance_name"), "외모 프리셋명")
        outfit_name = _safe_registration_name(payload.get("outfit_name"), "복장 프리셋명")
        if not session["fields"]["appearance"]:
            raise CharacterMakerError("외모 태그를 하나 이상 준비한 뒤 확정하세요.")
        if not session["fields"]["outfit"]:
            raise CharacterMakerError("복장 태그를 하나 이상 준비한 뒤 확정하세요.")

        expression_mode = str(payload.get("expression_mode") or "none")
        if expression_mode not in ("none", "existing", "new"):
            raise CharacterMakerError("표정 등록 방식이 올바르지 않습니다.")
        expression_name = ""
        if expression_mode in ("existing", "new"):
            expression_name = _safe_registration_name(
                payload.get("expression_name"), "표정 프리셋명"
            )
        if expression_mode == "new" and not session["fields"]["expression"]:
            raise CharacterMakerError(
                "새 표정 프리셋을 등록하려면 표정 태그를 하나 이상 준비하세요."
            )

        composition_mode = str(payload.get("composition_mode") or "none")
        if composition_mode not in ("none", "new"):
            raise CharacterMakerError("구도 등록 방식이 올바르지 않습니다.")
        composition_name = ""
        if composition_mode == "new":
            composition_name = _safe_registration_name(
                payload.get("composition_name"), "구도 프리셋명"
            )
            if not session["fields"]["composition"]:
                raise CharacterMakerError(
                    "새 구도 프리셋을 등록하려면 구도/기타 태그를 하나 이상 준비하세요."
                )

        asset_mode_module = importlib.import_module("modes.asset_mode")
        tags_file = asset_mode_module.TAGS_FILE
        asset_dir = asset_mode_module.ASSET_DIR
        requirements_dir = asset_mode_module.NAME_MAPPING_BACKUP_DIR
        old_tags = self.asset_manager.get_tags()
        new_tags = copy.deepcopy(old_tags)

        collisions: list[str] = []
        if character_name in new_tags.get("characters", {}):
            collisions.append(f"캐릭터 '{character_name}'")
        if appearance_name in new_tags.get("appearances", {}):
            collisions.append(f"외모 '{appearance_name}'")
        if outfit_name in new_tags.get("outfits", {}):
            collisions.append(f"복장 '{outfit_name}'")
        if expression_mode == "new" and expression_name in new_tags.get("expressions", {}):
            collisions.append(f"표정 '{expression_name}'")
        if expression_mode == "existing" and expression_name not in new_tags.get("expressions", {}):
            raise CharacterMakerError(f"기존 표정 프리셋 '{expression_name}'을 찾을 수 없습니다.")
        if (
            composition_mode == "new"
            and composition_name in new_tags.get("composition_presets", {})
        ):
            collisions.append(f"구도 '{composition_name}'")
        if collisions:
            print(
                f"[CHARACTER_MAKER] 확정 충돌: session={session_id}, "
                f"collisions={collisions}"
            )
            raise CharacterMakerError("이미 존재하는 이름입니다: " + ", ".join(collisions))

        new_tags.setdefault("appearances", {})[appearance_name] = list(
            session["fields"]["appearance"]
        )
        new_tags.setdefault("outfits", {})[outfit_name] = list(
            session["fields"]["outfit"]
        )
        if expression_mode == "new":
            new_tags.setdefault("expressions", {})[expression_name] = list(
                session["fields"]["expression"]
            )
        if composition_mode == "new":
            new_tags.setdefault("composition_presets", {})[composition_name] = list(
                session["fields"]["composition"]
            )
        new_tags.setdefault("characters", {})[character_name] = {
            "appearance": appearance_name,
            "outfit": outfit_name,
            "expression": expression_name,
        }

        revision_id = (
            str(payload.get("revision_id") or "")
            if "revision_id" in payload
            else str(session.get("active_revision_id") or "")
        )
        promote_revision = None
        if revision_id:
            promote_revision = next(
                (item for item in session["revisions"] if item["id"] == revision_id),
                None,
            )
            if promote_revision is None:
                raise CharacterMakerError("등록할 대표 이미지 리비전을 찾을 수 없습니다.")
        else:
            print(
                f"[CHARACTER_MAKER] 대표 이미지 승격 생략: "
                f"session={session_id}, revision_id 없음"
            )
        if promote_revision is not None and not expression_name:
            raise CharacterMakerError(
                "대표 이미지를 등록하려면 기존 또는 새 표정 프리셋을 선택하세요."
            )

        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        os.makedirs(requirements_dir, exist_ok=True)
        backup_path = os.path.join(
            requirements_dir, f"tags_before_character_maker_{stamp}.json"
        )
        if os.path.isfile(tags_file):
            shutil.copy2(tags_file, backup_path)
            print(
                f"[CHARACTER_MAKER] tags.json 백업 완료: "
                f"session={session_id}, backup={backup_path}"
            )
        else:
            with open(backup_path, "w", encoding="utf-8") as handle:
                json.dump(old_tags, handle, ensure_ascii=False, indent=2)
            print(
                f"[CHARACTER_MAKER] 메모리 태그 백업 완료: "
                f"session={session_id}, backup={backup_path}"
            )

        char_dir = os.path.join(asset_dir, self.asset_manager._safe_dirname(character_name))
        char_dir_existed = os.path.exists(char_dir)
        if char_dir_existed:
            raise CharacterMakerError(
                "동일한 저장 폴더가 이미 존재합니다. 다른 캐릭터명을 사용하세요."
            )

        promoted_image = ""
        try:
            self._atomic_write_json(tags_file, new_tags)
            self.asset_manager._tags = copy.deepcopy(new_tags)
            self.asset_manager._tags_loaded = True

            if promote_revision is not None:
                destination = os.path.join(
                    char_dir,
                    self.asset_manager._safe_dirname(outfit_name),
                    self.asset_manager._safe_dirname(expression_name),
                )
                os.makedirs(destination, exist_ok=True)
                source_image = promote_revision["image_path"]
                source_prompt = promote_revision["prompt_path"]
                image_name = os.path.basename(source_image)
                prompt_name = os.path.basename(source_prompt)
                image_target = os.path.join(destination, image_name)
                prompt_target = os.path.join(destination, prompt_name)
                shutil.copy2(source_image, image_target)
                prompt_payload = {
                    "positive": promote_revision["positive"],
                    "negative": promote_revision["negative"],
                    "character": character_name,
                    "appearance": appearance_name,
                    "outfit": outfit_name,
                    "expression": expression_name,
                    "character_maker_fields": copy.deepcopy(session["fields"]),
                    "composition_preset": composition_name,
                    "source_revision": promote_revision["id"],
                }
                self._atomic_write_json(prompt_target, prompt_payload)
                self._atomic_write_json(
                    os.path.join(destination, "_representative.json"),
                    {"filename": image_name},
                )
                promoted_image = image_target
        except Exception as exc:
            print(
                f"[CHARACTER_MAKER] 확정 저장 실패, 태그 롤백: "
                f"session={session_id}, error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            try:
                self._atomic_write_json(tags_file, old_tags)
                self.asset_manager._tags = copy.deepcopy(old_tags)
                self.asset_manager._tags_loaded = True
            except Exception as rollback_exc:
                print(
                    f"[CHARACTER_MAKER] tags.json 롤백 실패: "
                    f"session={session_id}, error={rollback_exc}, backup={backup_path}"
                )
                traceback.print_exc()
            if not char_dir_existed and os.path.isdir(char_dir):
                try:
                    asset_root_real = os.path.realpath(asset_dir)
                    char_real = os.path.realpath(char_dir)
                    if (
                        os.path.commonpath([asset_root_real, char_real]) == asset_root_real
                        and char_real != asset_root_real
                    ):
                        shutil.rmtree(char_real)
                except Exception as cleanup_exc:
                    print(
                        f"[CHARACTER_MAKER] 실패한 승격 폴더 정리 실패: "
                        f"path={char_dir}, error={cleanup_exc}"
                    )
                    traceback.print_exc()
            raise CharacterMakerError(f"캐릭터 확정 저장 실패: {exc}") from exc

        finalized = {
            "at": _now_iso(),
            "character_name": character_name,
            "appearance_name": appearance_name,
            "outfit_name": outfit_name,
            "expression_name": expression_name,
            "composition_name": composition_name,
            "revision_id": revision_id,
            "promoted_image": bool(promoted_image),
            "backup_path": backup_path,
        }
        session["finalized"] = finalized
        session["updated_at"] = _now_iso()
        print(
            f"[CHARACTER_MAKER] 캐릭터 확정 완료: session={session_id}, "
            f"character={character_name!r}, image={bool(promoted_image)}"
        )
        return {
            "success": True,
            "finalized": finalized,
            "session": self.public_session(session_id),
        }

    def delete_session(self, session_id: str) -> None:
        session = self._session(session_id)
        session_dir = os.path.join(self.temp_root, session["id"])
        _assert_within(self.temp_root, session_dir)
        if os.path.isdir(session_dir):
            shutil.rmtree(session_dir)
        del self.sessions[session_id]
        self._operation_locks.pop(session_id, None)
        print(f"[CHARACTER_MAKER] 임시 세션 삭제 완료: session={session_id}")
