"""Per-character, persistent maker proposals and non-destructive version restores."""
from __future__ import annotations

import asyncio
import copy
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
import shutil
import uuid
from urllib.parse import urlencode

from aiohttp import web
from PIL import Image

from asset_json import join_moss_caption_parts
from asset_preset_scope import write_scope_documents
from configuration import settings_lock
from json_store import write_json_atomic

MAX_UPLOAD_IMAGES = 6
MAX_UPLOAD_IMAGE_BYTES = 12 * 1024 * 1024
MAX_UPLOAD_TOTAL_BYTES = 36 * 1024 * 1024
UPLOAD_IMAGE_FORMATS = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp"}
MAKER_MODEL_FAMILIES = ("anima", "moss")
MAKER_REASONING_DEPTHS = ("fast", "deep")


def normalize_maker_model_family(value):
    if value is None or value == "":
        return ""
    if not isinstance(value, str):
        raise ValueError("메이커 모드는 Anima 또는 MOSS만 선택할 수 있습니다.")
    normalized = value.strip().lower()
    if normalized not in MAKER_MODEL_FAMILIES:
        raise ValueError("메이커 모드는 Anima 또는 MOSS만 선택할 수 있습니다.")
    return normalized


def normalize_maker_reasoning_depth(value):
    if value is None or value == "":
        return "fast"
    if not isinstance(value, str):
        raise ValueError("추론 레벨은 빠르게 또는 깊게만 선택할 수 있습니다.")
    normalized = value.strip().lower()
    if normalized not in MAKER_REASONING_DEPTHS:
        raise ValueError("추론 레벨은 빠르게 또는 깊게만 선택할 수 있습니다.")
    return normalized


def _merge_moss_prose_items(items):
    cleaned = [item.strip() for item in items if isinstance(item, str) and item.strip()]
    if len(cleaned) <= 1:
        return cleaned
    merged = [cleaned[0]]
    for item in cleaned[1:]:
        if merged[-1].endswith((".", "!", "?")):
            merged[-1] = f"{merged[-1]} {item}"
        else:
            merged.append(item)
    return merged


def validate_uploaded_images(uploaded_files):
    if not uploaded_files:
        return []
    if not isinstance(uploaded_files, list):
        raise ValueError("첨부 이미지 형식을 확인하세요.")
    if len(uploaded_files) > MAX_UPLOAD_IMAGES:
        raise ValueError(f"이미지는 한 번에 최대 {MAX_UPLOAD_IMAGES}장까지 첨부할 수 있습니다.")
    validated = []
    total_bytes = 0
    for index, item in enumerate(uploaded_files, start=1):
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError("첨부 이미지 형식을 확인하세요.")
        filename, data = item
        if not isinstance(data, (bytes, bytearray)) or not data:
            raise ValueError("비어 있는 이미지 파일은 첨부할 수 없습니다.")
        raw = bytes(data)
        if len(raw) > MAX_UPLOAD_IMAGE_BYTES:
            raise ValueError("이미지 한 장은 12MB를 넘을 수 없습니다.")
        total_bytes += len(raw)
        if total_bytes > MAX_UPLOAD_TOTAL_BYTES:
            raise ValueError("첨부한 이미지의 전체 크기는 36MB를 넘을 수 없습니다.")
        try:
            with Image.open(BytesIO(raw)) as opened:
                fmt = (opened.format or "").upper()
                if fmt not in UPLOAD_IMAGE_FORMATS:
                    raise ValueError("PNG, JPEG, WebP 이미지 파일을 선택하세요.")
                opened.load()
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("이미지 파일을 읽을 수 없습니다.") from exc
        ext = UPLOAD_IMAGE_FORMATS[fmt]
        clean_name = Path(str(filename or "").replace("\\", "/")).name.strip()[:200] or f"image-{index}{ext}"
        validated.append((clean_name, ext, raw))
    return validated


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def validate_settings(value):
    if (not isinstance(value, dict) or set(value) != {"defaults", "values"}
            or not all(isinstance(part, dict) for part in value.values())):
        raise ValueError("캐릭터 기본값과 적용값을 확인하세요.")
    if len(json.dumps(value, ensure_ascii=False, allow_nan=False)) > 2_000_000:
        raise ValueError("캐릭터 설정이 너무 큽니다.")
    for part in value.values():
        for key, tags in part.items():
            if key.startswith("raw_") and key.endswith("_tags") and (
                    not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags)):
                raise ValueError("태그는 문자열 목록이어야 합니다.")
        if "natural_language" in part and not isinstance(part["natural_language"], str):
            raise ValueError("자연어는 문자열이어야 합니다.")
    return copy.deepcopy(value)


class AssetCharacterMaker:
    def __init__(self, characters, root):
        self.characters = characters
        self.root = Path(root)
        self.running = set()

    def path(self, name):
        # Names never become filesystem paths.
        return self.root / f"{fingerprint(name)}.json"

    def uploads_dir(self, name):
        return self.root / "uploads" / fingerprint(name)

    @staticmethod
    def _uploaded_images_by_id(state):
        known = {}
        pending = state.get("pending")
        if isinstance(pending, dict):
            for item in pending.get("uploaded_images") or []:
                if isinstance(item, dict) and item.get("id"):
                    known[item["id"]] = item
        for version in state.get("versions") or []:
            if isinstance(version, dict):
                for item in version.get("uploaded_images") or []:
                    if isinstance(item, dict) and item.get("id"):
                        known[item["id"]] = item
        return known

    def _remove_uploaded_files(self, name, items):
        if not items:
            return
        uploads_root = self.uploads_dir(name).resolve()
        for item in items:
            raw_path = item.get("path", "") if isinstance(item, dict) else ""
            if not raw_path:
                continue
            try:
                resolved = Path(raw_path).resolve()
                if resolved.parent == uploads_root and resolved.is_file():
                    resolved.unlink(missing_ok=True)
            except OSError:
                pass
        try:
            if uploads_root.is_dir() and not any(uploads_root.iterdir()):
                uploads_root.rmdir()
        except OSError:
            pass

    def _clear_uploads_dir(self, name):
        uploads_root = self.uploads_dir(name)
        if uploads_root.is_dir():
            shutil.rmtree(uploads_root, ignore_errors=True)

    @staticmethod
    def _public_uploads(name, items):
        public = []
        for item in items or []:
            if not isinstance(item, dict) or not item.get("id"):
                continue
            public.append({
                "id": item["id"],
                "name": str(item.get("name") or ""),
                "url": "/api/asset/asset/character-maker/image?" + urlencode({
                    "name": name, "upload_id": item["id"],
                }),
            })
        return public

    def _character(self, name):
        if not isinstance(name, str) or not name:
            raise ValueError("캐릭터를 선택하세요.")
        catalog = self.characters.catalog()
        if name not in catalog["characters"]:
            raise ValueError("캐릭터를 찾을 수 없습니다. 다시 선택하세요.")
        return catalog, catalog["characters"][name]

    def _read(self, name):
        path = self.path(name)
        state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
            "name": name, "versions": [], "active_version_id": "", "pending": None, "error": "",
        }
        if state.get("name") != name or not isinstance(state.get("versions"), list):
            raise ValueError("캐릭터 버전 파일을 확인하세요.")
        if state.get("pending") and state["pending"]["id"] not in self.running:
            saved_upload_ids = {
                item.get("id")
                for version in state.get("versions") or []
                if isinstance(version, dict)
                for item in version.get("uploaded_images") or []
                if isinstance(item, dict) and item.get("id")
            }
            orphan_uploads = [
                item for item in (state["pending"].get("uploaded_images") or [])
                if isinstance(item, dict) and item.get("id") not in saved_upload_ids
            ]
            state.update(pending=None, error="서버 재시작으로 AI 요청이 중단되었습니다. 다시 요청하세요.")
            write_json_atomic(path, state)
            self._remove_uploaded_files(name, orphan_uploads)
        return state

    def _public(self, state, character):
        result = {**copy.deepcopy(state), "revision": fingerprint(state), "character_revision": fingerprint(character)}
        for version in result["versions"]:
            path = version.pop("image_path", "")
            if path:
                version["image_url"] = "/api/asset/asset/character-maker/image?" + urlencode({
                    "name": state["name"], "version_id": version["id"], "updated": version.get("image_updated_at", ""),
                })
            raw_uploads = version.pop("uploaded_images", None)
            if raw_uploads:
                version["uploaded_images"] = self._public_uploads(state["name"], raw_uploads)
        if isinstance(result.get("pending"), dict):
            raw_uploads = result["pending"].pop("uploaded_images", None)
            if raw_uploads:
                result["pending"]["uploaded_images"] = self._public_uploads(state["name"], raw_uploads)
        return result

    def image_path(self, name, version_id):
        with settings_lock(self.characters.path):
            self._character(name)
            version = next((v for v in self._read(name)["versions"] if v["id"] == version_id), None)
            path = version.get("image_path", "") if version else ""
            if not path or not Path(path).is_file():
                raise ValueError("이 버전의 생성 이미지를 찾을 수 없습니다. 다시 생성하세요.")
            return path

    def uploaded_image_entries(self, name, items):
        with settings_lock(self.characters.path):
            self._character(name)
            state = self._read(name)
            known = self._uploaded_images_by_id(state)
            uploads_root = self.uploads_dir(name).resolve()
            entries = []
            for item in items or []:
                upload_id = item.get("id") if isinstance(item, dict) else str(item or "")
                record = known.get(upload_id)
                path = record.get("path", "") if record else ""
                if not path:
                    raise ValueError("첨부한 이미지를 찾을 수 없습니다. 다시 업로드해 주세요.")
                resolved = Path(path).resolve()
                if resolved.parent != uploads_root or not resolved.is_file():
                    raise ValueError("첨부한 이미지를 찾을 수 없습니다. 다시 업로드해 주세요.")
                entries.append({"id": upload_id, "name": str(record.get("name") or ""), "path": str(resolved)})
            return entries

    def uploaded_image_path(self, name, upload_id):
        entries = self.uploaded_image_entries(name, [{"id": upload_id}])
        return entries[0]["path"]

    def read(self, name):
        with settings_lock(self.characters.path):
            _, character = self._character(name)
            return self._public(self._read(name), character)

    def _checked(self, body):
        name = body.get("name")
        catalog, character = self._character(name)
        state = self._read(name)
        if body.get("revision") != fingerprint(state) or body.get("character_revision") != fingerprint(character):
            raise ValueError("캐릭터 또는 버전이 변경되었습니다. 새로고침 후 다시 시도하세요.")
        if state.get("pending"):
            raise ValueError("AI 결과를 기다린 뒤 다시 시도하세요.")
        settings = validate_settings(body.get("settings"))
        # Detect stale local drafts even if the caller has fetched a newer revision.
        saved = character.get("settings")
        if saved:
            # Background now belongs to style settings. Archived character
            # settings may still contain it; it cannot make a current draft stale.
            style_keys = {"background_preset", "raw_background_tags"}
            if any(settings[part].get(key) != value for part in ("defaults", "values")
                   for key, value in saved[part].items() if key not in style_keys):
                raise ValueError("화면과 저장된 적용값이 다릅니다. 캐릭터를 다시 선택하세요.")
        elif any(settings["values"].get(key, "") != character.get(key, "")
                 for key in ("appearance", "outfit", "expression")):
            raise ValueError("캐릭터 적용값을 먼저 저장하세요.")
        return name, catalog, character, state, settings

    @staticmethod
    def _append(state, kind, before, after, **extra):
        version = {
            "id": uuid.uuid4().hex, "number": len(state["versions"]) + 1, "kind": kind,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "before": copy.deepcopy(before), "after": copy.deepcopy(after), **extra,
        }
        state["versions"].append(version)
        return version

    def _checkpoint(self, state, settings):
        active = next((v for v in state["versions"] if v["id"] == state["active_version_id"]), None)
        if active and active["after"] == settings:
            return active
        version = self._append(state, "snapshot", settings, settings, message="현재 적용값 보관")
        state["active_version_id"] = version["id"]
        return version

    def change(self, body):
        with settings_lock(self.characters.path):
            name, catalog, character, state, settings = self._checked(body)
            action = body.get("action")
            if action == "snapshot":
                self._checkpoint(state, settings)
                state["error"] = ""
                write_json_atomic(self.path(name), state)
            elif action == "reset":
                state.update(versions=[], active_version_id="", pending=None, error="")
                write_json_atomic(self.path(name), state)
                self._clear_uploads_dir(name)
            elif action in {"apply", "rollback"}:
                target = next((v for v in state["versions"] if v["id"] == body.get("version_id")), None)
                if not target:
                    raise ValueError("선택한 버전을 찾을 수 없습니다.")
                if action == "apply" and target["kind"] != "proposal":
                    raise ValueError("적용할 AI 제안을 선택하세요.")
                scope = catalog["scopes"][name]
                if scope == "builtin":
                    self.characters.lock.require(body)
                after = validate_settings(target["after"])
                self._checkpoint(state, settings)
                version = self._append(state, action, settings, after, source_id=target["id"],
                                       message=f"v{target['number']} {'적용' if action == 'apply' else '롤백'}",
                                       **{key: target[key] for key in ("image_path", "image_updated_at") if key in target})
                state.update(active_version_id=version["id"], error="")
                path = self.characters.builtin_file if scope == "builtin" else self.characters.path
                document = self.characters._read_file(path)
                updated = document["characters"][name]
                updated["settings"] = after
                updated.update({key: after["values"].get(key, "") for key in ("appearance", "outfit", "expression")})
                # Roll back both documents if either write fails.
                write_scope_documents({self.path(name): state, path: document})
                character = updated
            else:
                raise ValueError("지원하지 않는 메이커 요청입니다.")
            return self._public(state, character)

    def begin(self, body, uploaded_files=None):
        validated_uploads = validate_uploaded_images(uploaded_files)
        with settings_lock(self.characters.path):
            name, _, character, state, settings = self._checked(body)
            generating = body.get("action") == "generate"
            if generating and validated_uploads:
                raise ValueError("이미지는 AI 대화 요청에만 첨부할 수 있습니다.")
            message = "이미지 생성" if generating else body.get("message")
            if not isinstance(message, str) or not message.strip() or len(message) > 8000:
                raise ValueError("AI 요청을 1~8,000자로 입력하세요.")
            base_id = body.get("base_id")
            if base_id:
                base = next((v for v in state["versions"] if v["id"] == base_id), None)
                if base is None:
                    raise ValueError("대화 기준 버전을 찾을 수 없습니다.")
            else:
                base = self._checkpoint(state, settings)
            # Context follows only the selected version's ancestry, including restores.
            ancestry, cursor, seen = [], base, set()
            by_id = {v["id"]: v for v in state["versions"]}
            while cursor and cursor["id"] not in seen and len(ancestry) < 6:
                seen.add(cursor["id"])
                if cursor["kind"] == "proposal":
                    ancestry.append(cursor)
                cursor = by_id.get(cursor.get("parent_id") or cursor.get("source_id"))
            history = [entry for version in reversed(ancestry) for entry in (
                {"role": "user", "content": version["message"], "accepted": True},
                {"role": "assistant", "content": version.get("assistant_message", ""), "accepted": True},
            )]
            pending = {"id": uuid.uuid4().hex, "message": message.strip(), "base_id": base["id"],
                       "phase": "generate" if generating else "propose"}
            model_family = normalize_maker_model_family(body.get("model_family"))
            reasoning_depth = normalize_maker_reasoning_depth(body.get("reasoning_depth"))
            if not generating and model_family:
                pending["model_family"] = model_family
            if not generating:
                pending["reasoning_depth"] = reasoning_depth
                if reasoning_depth == "deep":
                    pending["deep_pass"] = 1
                    pending["stage_label"] = (
                        "영역 탐지 준비 중…" if validated_uploads else "1차 상세 프롬프트 작성 중…"
                    )
            if generating:
                pending["version_id"] = base["id"]
            elif body.get("use_image", True) and base.get("image_path"):
                if not Path(base["image_path"]).is_file():
                    raise ValueError("기준 이미지가 없습니다. 다시 생성하거나 이미지 전달을 꺼 주세요.")
                pending["image_version_id"] = base["id"]
            saved_uploads = []
            try:
                if validated_uploads:
                    uploads_root = self.uploads_dir(name)
                    uploads_root.mkdir(parents=True, exist_ok=True)
                    for clean_name, ext, raw in validated_uploads:
                        upload_id = uuid.uuid4().hex
                        file_path = uploads_root / f"{upload_id}{ext}"
                        file_path.write_bytes(raw)
                        saved_uploads.append({"id": upload_id, "name": clean_name, "path": str(file_path)})
                    pending["uploaded_images"] = saved_uploads
                state.update(pending=pending, error="")
                self.running.add(pending["id"])
                write_json_atomic(self.path(name), state)
            except Exception:
                self.running.discard(pending["id"])
                self._remove_uploaded_files(name, saved_uploads)
                raise
            return self._public(state, character), copy.deepcopy(base["after"]), history

    def update_pending_stage(self, name, request_id, stage_label, *, phase=None, deep_pass=None):
        with settings_lock(self.characters.path):
            state = self._read(name)
            pending = state.get("pending")
            if not pending or pending["id"] != request_id:
                return
            if stage_label:
                pending["stage_label"] = str(stage_label)
            if phase:
                pending["phase"] = phase
            if deep_pass is not None:
                pending["deep_pass"] = deep_pass
            write_json_atomic(self.path(name), state)

    def finish(
        self, name, request_id, before, result=None, error="", *,
        generate=False, deep_pass=None, auto_refined=False,
        message_override=None, parent_id_override=None,
        feedback_image_version_id_override=None, keep_uploaded_images=False,
        stage_label="",
    ):
        with settings_lock(self.characters.path):
            state = self._read(name)
            pending = state.get("pending")
            if not pending or pending["id"] != request_id:
                return
            orphan_uploads = []
            if result is not None:
                after = copy.deepcopy(before)
                model_family = pending.get("model_family", "")
                reasoning_depth = pending.get("reasoning_depth", "fast")
                for key in ("appearance", "outfit", "expression", "composition"):
                    tags = result.get("fields", {}).get(key)
                    if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
                        raise ValueError("AI 결과 태그 형식을 확인하세요.")
                    after["values"][f"raw_{key}_tags"] = (
                        _merge_moss_prose_items(tags) if model_family == "moss" else tags
                    )
                # Expression and composition share one visible field in character settings.
                merged_expression = list(dict.fromkeys([
                    *after["values"]["raw_expression_tags"],
                    *after["values"].pop("raw_composition_tags"),
                ]))
                after["values"]["raw_expression_tags"] = (
                    _merge_moss_prose_items(merged_expression)
                    if model_family == "moss"
                    else merged_expression
                )
                after["values"].pop("composition_preset", None)
                if result.get("natural_language") is not None:
                    after["values"]["natural_language"] = result["natural_language"]
                validate_settings(after)
                if keep_uploaded_images:
                    uploaded_images = copy.deepcopy(pending.get("uploaded_images") or [])
                else:
                    uploaded_images = copy.deepcopy(pending.pop("uploaded_images", None) or [])
                extra = {
                    "feedback_image_version_id": (
                        feedback_image_version_id_override
                        if feedback_image_version_id_override is not None
                        else pending.get("image_version_id", "")
                    )
                }
                if model_family:
                    extra["model_family"] = model_family
                if reasoning_depth == "deep":
                    extra["reasoning_depth"] = "deep"
                if deep_pass is not None:
                    extra["deep_pass"] = deep_pass
                if auto_refined:
                    extra["auto_refined"] = True
                if uploaded_images and not auto_refined:
                    extra["uploaded_images"] = uploaded_images
                version = self._append(
                    state, "proposal", before, after,
                    parent_id=parent_id_override or pending["base_id"],
                    message=message_override if message_override is not None else pending["message"],
                    assistant_message=str(result.get("assistant_message", "")),
                    **extra,
                )
                if generate:
                    pending.update(phase="generate", version_id=version["id"])
                    if stage_label:
                        pending["stage_label"] = stage_label
                    if deep_pass is not None:
                        pending["deep_pass"] = deep_pass
            else:
                saved_upload_ids = {
                    item.get("id")
                    for version in state.get("versions") or []
                    if isinstance(version, dict)
                    for item in version.get("uploaded_images") or []
                    if isinstance(item, dict) and item.get("id")
                }
                orphan_uploads = [
                    item for item in (pending.get("uploaded_images") or [])
                    if isinstance(item, dict) and item.get("id") not in saved_upload_ids
                ]
            if error and pending.get("version_id"):
                target = next(v for v in state["versions"] if v["id"] == pending["version_id"])
                target["image_error"] = error
            state.update(pending=pending if generate and not error else None, error=error)
            write_json_atomic(self.path(name), state)
            if orphan_uploads:
                self._remove_uploaded_files(name, orphan_uploads)
            if not state["pending"]:
                self.running.discard(request_id)
            return copy.deepcopy(version) if result is not None else None

    def finish_image(self, name, request_id, result, *, next_stage_label="", next_deep_pass=None):
        with settings_lock(self.characters.path):
            state = self._read(name)
            pending = state.get("pending")
            if not pending or pending["id"] != request_id:
                return None, None
            path = result.get("local_path") if isinstance(result, dict) and result.get("success") else None
            if not path or not Path(path).is_file():
                raise ValueError((result.get("error") if isinstance(result, dict) else "") or "생성 이미지를 저장하지 못했습니다.")
            version = next(v for v in state["versions"] if v["id"] == pending["version_id"])
            version.update(image_path=str(path), image_updated_at=datetime.now(timezone.utc).isoformat(), image_error="")
            if next_stage_label:
                pending.update(
                    phase="propose",
                    stage_label=str(next_stage_label),
                    version_id="",
                    image_version_id=version["id"],
                )
                if next_deep_pass is not None:
                    pending["deep_pass"] = next_deep_pass
                state.update(pending=pending, error="")
            else:
                state.update(pending=None, error="")
                self.running.discard(request_id)
            write_json_atomic(self.path(name), state)
            return str(path), copy.deepcopy(version)


def generation_job(template, values):
    """The template contains style/model/reference settings, with editable text left empty."""
    from illustration_json import is_moss_document
    job = copy.deepcopy(template)
    tags = [tag for key in ("appearance", "expression", "composition", "outfit")
            for tag in values.get(f"raw_{key}_tags", [])]
    if is_moss_document(job.get("prompt_json")):
        from modes.moss_prompt_builder import MossPromptBuilder
        moss = job["prompt_json"]["moss"]
        character_tags = join_moss_caption_parts(tags)
        natural_language = str(values.get("natural_language") or "").strip()
        items = moss.get("items")
        if isinstance(items, list) and len(items) == 1 and isinstance(items[0], dict):
            existing = str(items[0].get("caption") or "").strip()
            if existing == "character":
                prefix = ""
            elif existing.endswith(", character"):
                prefix = existing[:-len(", character")].strip()
            else:
                prefix = existing
            body = join_moss_caption_parts((character_tags, natural_language))
            items[0]["caption"] = join_moss_caption_parts((prefix, body or "character"))
        else:
            summary = moss.setdefault("summary", {})
            parts = [summary.get("caption", ""), *tags, natural_language]
            summary["caption"] = join_moss_caption_parts(parts)
        job["positive_prompt"] = MossPromptBuilder.dumps(job["prompt_json"])
        return job
    fields = job["prompt_json"]["fields"]
    multi = fields.get("MULTI_CHAR", {})
    if multi.get("mask_area_limit_enabled"):
        character_tags = ", ".join(tag.strip() for tag in tags if isinstance(tag, str) and tag.strip())
        natural_language = values.get("natural_language", "")
        multi["char_inform"] = [character_tags or ("" if natural_language else "character")]
        multi["char_composition_list"] = [natural_language]
    for field in ("ANIMA_CONTENT", "SDXL_CONTENT"):
        if field not in fields:
            continue
        parts = [fields[field], *tags]
        if field == "ANIMA_CONTENT":
            parts.append(values.get("natural_language", ""))
        fields[field] = ", ".join(part.strip() for part in parts if isinstance(part, str) and part.strip())
    job["positive_prompt"] = json.dumps(job["prompt_json"], ensure_ascii=False, allow_nan=False)
    return job


def register_character_maker_runtime(server, prepare_generation=None):
    store = AssetCharacterMaker(server.asset_mode.character_presets,
                                Path(server.CONFIG_FILE).parent / "asset_character_versions")

    async def read_request(request):
        if request.content_type and request.content_type.startswith("multipart/"):
            reader = await request.multipart()
            payload_raw = ""
            uploaded_files = []
            total_bytes = 0
            async for part in reader:
                if part.name == "payload":
                    data = await part.read(decode=False)
                    if len(data) > 4_000_000:
                        raise ValueError("메이커 요청 크기가 너무 큽니다.")
                    payload_raw = data.decode("utf-8")
                elif part.name == "image":
                    if len(uploaded_files) >= MAX_UPLOAD_IMAGES:
                        raise ValueError(f"이미지는 한 번에 최대 {MAX_UPLOAD_IMAGES}장까지 첨부할 수 있습니다.")
                    chunks = []
                    size = 0
                    while True:
                        chunk = await part.read_chunk(64 * 1024)
                        if not chunk:
                            break
                        size += len(chunk)
                        if size > MAX_UPLOAD_IMAGE_BYTES:
                            raise ValueError("이미지 한 장은 12MB를 넘을 수 없습니다.")
                        if total_bytes + size > MAX_UPLOAD_TOTAL_BYTES:
                            raise ValueError("첨부한 이미지의 전체 크기는 36MB를 넘을 수 없습니다.")
                        chunks.append(chunk)
                    data = b"".join(chunks)
                    if data:
                        total_bytes += len(data)
                        uploaded_files.append((part.filename or "image.png", data))
            if not payload_raw:
                raise ValueError("메이커 요청을 확인하세요.")
            body = json.loads(payload_raw)
            return body, uploaded_files
        return await request.json(), []

    async def endpoint(request):
        try:
            if request.method == "GET":
                result = store.read(request.query.get("name", ""))
            else:
                body, uploaded_files = await read_request(request)
                if not isinstance(body, dict):
                    raise ValueError("메이커 요청을 확인하세요.")
                if uploaded_files and body.get("action") != "propose":
                    raise ValueError("이미지는 AI 대화 요청에만 첨부할 수 있습니다.")
                if uploaded_files:
                    validate_uploaded_images(uploaded_files)
                if body.get("action") in {"propose", "generate"}:
                    template = body.get("generation_job")
                    snapshot = None
                    if template is not None:
                        if not isinstance(template, dict) or prepare_generation is None:
                            raise ValueError("이미지 생성 설정을 확인하세요.")
                        template = copy.deepcopy(template)
                        template["character"] = body.get("name")
                        snapshot = prepare_generation(template)
                        template.update(storage_group="character_maker", storage_session=f"asset-{fingerprint(body.get('name'))}")
                    elif body.get("action") == "generate":
                        raise ValueError("이미지 생성 설정을 확인하세요.")
                    resolved_family = normalize_maker_model_family(body.get("model_family"))
                    if body.get("action") == "propose" and not resolved_family:
                        workflow_type = (
                            (snapshot or {}).get("config", {}).get("asset_workflow_type")
                            or body.get("asset_workflow_type")
                            or ""
                        )
                        if not workflow_type and hasattr(server, "app_config"):
                            try:
                                from asset_model_presets import AssetModelPresets
                                workflow_type = AssetModelPresets(server.CONFIG_FILE).snapshot(server.app_config)["config"]["asset_workflow_type"]
                            except Exception:
                                workflow_type = ""
                        resolved_family = "moss" if str(workflow_type).strip().lower() == "moss" else "anima"
                    resolved_depth = normalize_maker_reasoning_depth(body.get("reasoning_depth"))
                    begin_body = (
                        {**body, "model_family": resolved_family, "reasoning_depth": resolved_depth}
                        if body.get("action") == "propose"
                        else body
                    )
                    result, before, history = store.begin(begin_body, uploaded_files=uploaded_files)
                    name, pending = result["name"], result["pending"]
                    request_id = pending["id"]
                    uploaded_refs_snapshot = copy.deepcopy(pending.get("uploaded_images") or [])
                    detected_bbox_regions = []

                    def set_stage(stage_label):
                        store.update_pending_stage(name, request_id, stage_label)

                    async def enqueue_image(values, *, label_suffix="", next_stage_label="", next_deep_pass=None, on_done=None):
                        item = await server.queue_manager.add_item(
                            "asset_generation",
                            label=f"캐릭터 메이커 이미지{label_suffix} · {name}",
                            params={"body": generation_job(template, values), "asset_model_preset": snapshot},
                        )
                        def image_completed(future):
                            try:
                                saved_path, saved_version = store.finish_image(
                                    name, request_id, future.result(),
                                    next_stage_label=next_stage_label,
                                    next_deep_pass=next_deep_pass,
                                )
                            except BaseException as exc:
                                store.finish(
                                    name, request_id, before,
                                    error=str(exc) or "이미지 생성이 취소되었습니다. 다시 생성하세요.",
                                )
                                return
                            if on_done and saved_path and saved_version:
                                asyncio.ensure_future(on_done(saved_path, saved_version))
                        item.completion_future.add_done_callback(image_completed)

                    async def schedule_deep_pass2(image1_path, version1):
                        async def run_pass2(_item):
                            ref_entries = (
                                store.uploaded_image_entries(name, uploaded_refs_snapshot)
                                if uploaded_refs_snapshot
                                else None
                            )
                            orig_msg = pending["message"]
                            if ref_entries:
                                pass2_message = (
                                    f"Deep refinement pass (2/2) for the user's request: {orig_msg}\n"
                                    "Compare GENERATED_CURRENT carefully against USER_UPLOADED_* and all "
                                    "*_CROP_* close-ups. Identify EVERY concrete discrepancy (hairstyle "
                                    "count/bangs, headpiece, left/right asymmetry, mantle/cape ending "
                                    "length, collar, bodice color panels, sleeves/gloves, hip ornaments, "
                                    "skirt shape/pattern, overskirt length, legwear, footwear height, "
                                    "held prop hand/length/shaft color/finial, leaked background) and "
                                    "produce a corrected prompt that fixes every mismatch."
                                )
                            else:
                                pass2_message = (
                                    f"Deep self-review pass (2/2) for the user's request: {orig_msg}\n"
                                    "Inspect GENERATED_CURRENT against the user's request and current "
                                    "fields. Identify any missing, drifted, or contradictory details "
                                    "and produce an improved prompt."
                                )
                            pass2_history = [
                                *history,
                                {"role": "user", "content": orig_msg, "accepted": True},
                                {"role": "assistant", "content": str(version1.get("assistant_message") or ""), "accepted": True},
                            ]
                            pass2_kwargs = {
                                "image_path": image1_path,
                                "model_family": resolved_family,
                                "reasoning_depth": "deep",
                                "deep_pass": 2,
                                "precomputed_bbox_regions": detected_bbox_regions if ref_entries else None,
                                "progress_callback": set_stage,
                            }
                            if ref_entries:
                                pass2_kwargs["reference_images"] = ref_entries
                            proposal2 = await server.character_maker.suggest_asset_values(
                                name, version1["after"]["values"], pass2_message, pass2_history, **pass2_kwargs,
                            )
                            version2 = store.finish(
                                name, request_id, version1["after"], proposal2,
                                generate=True,
                                deep_pass=2,
                                auto_refined=True,
                                message_override="2차 자동 디테일 보정",
                                parent_id_override=version1["id"],
                                feedback_image_version_id_override=version1["id"],
                                stage_label="2차 이미지 생성 중…",
                            )
                            if version2:
                                await enqueue_image(
                                    version2["after"]["values"],
                                    label_suffix=" (2차 보정)",
                                )
                            return {"success": True}

                        def pass2_completed(future):
                            try:
                                future.result()
                            except BaseException as exc:
                                store.finish(
                                    name, request_id, version1["after"],
                                    error=str(exc) or "2차 디테일 보정이 취소되었습니다. 1차 결과는 보관되었습니다.",
                                )

                        try:
                            item2 = await server.queue_manager.add_item(
                                "character_maker",
                                label=f"캐릭터 메이커 2차 보정 · {name}",
                                params={"character": name},
                                runtime_handler=run_pass2,
                            )
                            item2.completion_future.add_done_callback(pass2_completed)
                        except BaseException as exc:
                            store.finish(
                                name, request_id, version1["after"],
                                error=str(exc) or "2차 디테일 보정을 시작하지 못했습니다. 1차 결과는 보관되었습니다.",
                            )

                    async def run(_item):
                        image_path = store.image_path(name, pending["image_version_id"]) if pending.get("image_version_id") else None
                        kwargs = {"image_path": image_path, "model_family": resolved_family}
                        if uploaded_refs_snapshot:
                            kwargs["reference_images"] = store.uploaded_image_entries(name, uploaded_refs_snapshot)
                        if resolved_depth == "deep":
                            kwargs["reasoning_depth"] = "deep"
                            kwargs["deep_pass"] = 1
                            kwargs["bbox_regions_out"] = detected_bbox_regions
                            kwargs["progress_callback"] = set_stage
                        proposal = await server.character_maker.suggest_asset_values(
                            name, before["values"], pending["message"], history, **kwargs,
                        )
                        will_run_pass2 = resolved_depth == "deep" and template is not None
                        version = store.finish(
                            name, request_id, before, proposal,
                            generate=template is not None,
                            deep_pass=1 if resolved_depth == "deep" else None,
                            keep_uploaded_images=will_run_pass2,
                            stage_label="1차 이미지 생성 중…" if will_run_pass2 else "",
                        )
                        if template is not None and version:
                            if will_run_pass2:
                                await enqueue_image(
                                    version["after"]["values"],
                                    label_suffix=" (1차)",
                                    next_stage_label="원본 대조·2차 디테일 보정 중…",
                                    next_deep_pass=2,
                                    on_done=schedule_deep_pass2,
                                )
                            else:
                                await enqueue_image(version["after"]["values"])
                        return {"success": True}

                    def completed(future):
                        try:
                            future.result()
                        except BaseException as exc:
                            store.finish(name, request_id, before,
                                         error=str(exc) or "AI 요청이 취소되었습니다. 다시 요청하세요.")

                    try:
                        if body.get("action") == "generate":
                            await enqueue_image(before["values"])
                        else:
                            item = await server.queue_manager.add_item(
                                "character_maker", label=f"캐릭터 메이커 · {name}",
                                params={"character": name}, runtime_handler=run,
                            )
                            item.completion_future.add_done_callback(completed)
                    except Exception as exc:
                        store.finish(name, request_id, before, error=str(exc))
                        raise
                else:
                    result = store.change(body)
                    if body.get("action") in {"apply", "rollback"}:
                        server.asset_mode.load_tags()
            return web.json_response(result, headers={"Cache-Control": "no-store"})
        except (ValueError, TypeError, OSError, KeyError) as exc:
            return web.json_response({"error": str(exc)}, status=400)

    server.app.router.add_get("/api/asset/character-maker", endpoint)
    server.app.router.add_post("/api/asset/character-maker", endpoint)

    async def image(request):
        try:
            name = request.query.get("name", "")
            upload_id = request.query.get("upload_id", "")
            path = (store.uploaded_image_path(name, upload_id) if upload_id
                    else store.image_path(name, request.query.get("version_id", "")))
            return web.FileResponse(path, headers={"Cache-Control": "no-store"})
        except (ValueError, OSError):
            raise web.HTTPNotFound()

    server.app.router.add_get("/api/asset/character-maker/image", image)
