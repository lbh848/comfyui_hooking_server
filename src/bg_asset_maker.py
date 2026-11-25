"""Per-theme persistent background maker proposals, LLM suggestions, and version restores.

Maker edits are strictly scoped to the selected place's base `tags` and `natural_language`.
Theme common tags, negative tags, LoRAs, other places, and all time-of-day entries remain untouched.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time
import traceback
import uuid
from urllib.parse import urlencode

from aiohttp import web

from asset_character_maker import (
    MAX_UPLOAD_IMAGES,
    MAX_UPLOAD_IMAGE_BYTES,
    MAX_UPLOAD_TOTAL_BYTES,
    fingerprint,
    validate_uploaded_images,
)
from asset_preset_scope import write_scope_documents
from bg_asset_presets import normalize_bg_theme
from configuration import settings_lock
from json_store import write_json_atomic
from modes import llm_prompt_edit, llm_service

_BG_MAKER_SYSTEM_PROMPT_ANIMA = (
    "You are an expert AI Background Place Designer helping a user create and refine the permanent "
    "environment description of a single background location for anime-style diffusion models "
    "(ANIMA / Illustrious XL).\n\n"
    "Your ONLY job is to output the target place's base 'tags' and 'natural_language'.\n"
    "Time-of-day lighting, sky color, and weather (day, morning, sunset, night, rain, clouds, dawn, etc.) "
    "are stored and applied SEPARATELY per time slot, so do NOT bake time-of-day or weather into the place fields.\n\n"
    "Strict rules:\n"
    "1. Return ONLY a single valid JSON object. Do not wrap in markdown fences or extra commentary.\n"
    "2. Focus exclusively on permanent physical architecture, scenery, furniture, props, terrain, "
    "spatial layout, and camera perspective of the target place.\n"
    "3. Do NOT include time-of-day, sky/sun/moon lighting, or weather tags unless they are permanent physical features.\n"
    "4. Do NOT include artist tags, quality tags (masterpiece, best quality, score_9, etc.), "
    "or characters/people (1girl, 1boy, crowd) unless the user explicitly asks for them.\n"
    "5. Write 'tags' as concise English Danbooru/booru-style scenery tags, and 'natural_language' as an optional "
    "concise English description of the place's physical layout and composition.\n"
    "6. Write 'assistant_message' in clear, concise Korean explaining what you created or changed for this place.\n"
    "7. When vision images are provided: [GENERATED_CURRENT] (blue top banner) is the current generated preview "
    "(which may include a separate time-of-day lighting layer) to inspect and refine the physical location; "
    "[USER_UPLOADED_N] (red top banner) are user reference images to draw architectural/prop inspiration from.\n\n"
    "Required JSON schema:\n"
    "{\n"
    '  "assistant_message": "Korean explanation of changes to the place",\n'
    '  "tags": ["english", "scenery", "tags", "for", "this", "place"],\n'
    '  "natural_language": "Concise English description of the physical place (or empty string if not needed)"\n'
    "}"
)

_BG_MAKER_SYSTEM_PROMPT_MOSS = (
    "You are an expert AI Background Place Designer helping a user create and refine the permanent "
    "environment description of a single background location for the MOSS natural-language image model.\n\n"
    "Your ONLY job is to output the target place's base 'tags' and 'natural_language'.\n"
    "Time-of-day lighting, sky color, and weather (day, morning, sunset, night, rain, clouds, dawn, etc.) "
    "are stored and applied SEPARATELY per time slot, so keep the place description lighting-neutral and "
    "do NOT bake a specific time of day or weather into the place fields.\n\n"
    "Strict rules:\n"
    "1. Return ONLY a single valid JSON object. Do not wrap in markdown fences or extra commentary.\n"
    "2. Focus exclusively on permanent physical architecture, scenery, materials, furniture, props, "
    "spatial layout, depth, and camera perspective of the target place.\n"
    "3. Do NOT include time-of-day, sky/sun/moon lighting, or weather unless explicitly requested as a permanent feature.\n"
    "4. Do NOT include artist names, quality tags, or foreground people/characters unless explicitly requested.\n"
    "5. Write 'tags' as concise English location/prop keywords and 'natural_language' as a vivid, concrete "
    "English paragraph describing the place's architecture, objects, and composition.\n"
    "6. Write 'assistant_message' in clear, concise Korean explaining what you created or changed for this place.\n"
    "7. When vision images are provided: [GENERATED_CURRENT] (blue top banner) is the current generated preview "
    "(rendered with a separate time-of-day layer) to inspect and refine the physical location; "
    "[USER_UPLOADED_N] (red top banner) are user reference images to draw architectural/prop inspiration from.\n\n"
    "Required JSON schema:\n"
    "{\n"
    '  "assistant_message": "Korean explanation of changes to the place",\n'
    '  "tags": ["concise", "english", "location", "keywords"],\n'
    '  "natural_language": "Vivid English description of the place architecture, props, and composition"\n'
    "}"
)


def _pick_focus(theme: dict, preferred_place: str = "", preferred_time: str = "") -> tuple[str, str]:
    places = theme.get("places") if isinstance(theme, dict) else {}
    if not isinstance(places, dict) or not places:
        return "", ""
    place = preferred_place if preferred_place in places else next(iter(places))
    times = places[place].get("times") if isinstance(places[place], dict) else {}
    if not isinstance(times, dict) or not times:
        return place, ""
    time_name = preferred_time if preferred_time in times else next(iter(times))
    return place, time_name


def _validate_theme_payload(value: object) -> dict:
    if not isinstance(value, dict):
        raise ValueError("배경 테마 설정을 확인하세요.")
    if len(json.dumps(value, ensure_ascii=False, allow_nan=False)) > 2_000_000:
        raise ValueError("배경 테마 설정이 너무 큽니다.")
    return normalize_bg_theme(value, strict=True)


def _normalize_place_tags(raw_tags: object) -> list[str]:
    if raw_tags is None:
        return []
    if isinstance(raw_tags, str):
        raw_list = [part.strip() for part in raw_tags.split(",") if part.strip()]
    elif isinstance(raw_tags, list):
        if any(not isinstance(item, str) for item in raw_tags):
            raise ValueError("tags 항목은 문자열이어야 합니다.")
        raw_list = [item.strip() for item in raw_tags if item.strip()]
    else:
        raise ValueError("tags 필드는 문자열 배열이어야 합니다.")
    deduped: list[str] = []
    seen: set[str] = set()
    for tag in raw_list:
        if len(tag) > 300:
            raise ValueError("태그 한 항목은 300자를 넘을 수 없습니다.")
        key = tag.casefold()
        if key not in seen:
            seen.add(key)
            deduped.append(tag)
    if len(deduped) > 160:
        raise ValueError("장소 태그는 160개를 넘을 수 없습니다.")
    return deduped


def _apply_place_proposal(before: dict, target_place: str, proposal: dict, focus_time: str = "") -> tuple[dict, str, str]:
    after = copy.deepcopy(normalize_bg_theme(before, strict=False))
    if not target_place or target_place not in after["places"]:
        raise ValueError("편집할 장소를 찾을 수 없습니다. 먼저 장소를 선택하세요.")
    place_obj = after["places"][target_place]
    place_obj["tags"] = _normalize_place_tags(proposal.get("tags"))
    nl = proposal.get("natural_language", "")
    if nl is None:
        nl = ""
    if not isinstance(nl, str):
        raise ValueError("natural_language 필드는 문자열이어야 합니다.")
    if len(nl) > 4000:
        raise ValueError("장소 자연어는 4,000자를 넘을 수 없습니다.")
    place_obj["natural_language"] = nl.strip()
    normalized = _validate_theme_payload(after)
    eff_place, eff_time = _pick_focus(normalized, target_place, focus_time)
    return normalized, eff_place, eff_time


def _parse_bg_maker_llm_payload(raw: str, before: dict, target_place: str, focus_time: str) -> tuple[dict | None, str]:
    try:
        parsed = llm_prompt_edit.parse_llm_json(raw)
    except Exception as exc:
        return None, f"LLM 응답을 JSON으로 파싱하지 못했습니다: {type(exc).__name__}: {exc}"
    if not isinstance(parsed, dict):
        return None, "LLM 응답이 JSON 객체가 아닙니다."
    assistant_message = parsed.get("assistant_message")
    if not isinstance(assistant_message, str) or not assistant_message.strip():
        return None, "assistant_message(문자열)가 없거나 비어 있습니다."
    if "tags" not in parsed:
        return None, "tags(문자열 배열) 필드가 없습니다."
    try:
        after, eff_place, eff_time = _apply_place_proposal(before, target_place, parsed, focus_time)
    except ValueError as exc:
        return None, str(exc)
    return {
        "assistant_message": assistant_message.strip(),
        "after": after,
        "focus_place": eff_place,
        "focus_time": eff_time,
    }, ""


async def suggest_bg_theme_values(
    character_maker,
    theme_name: str,
    before: dict,
    message: str,
    history: list[dict],
    *,
    focus_place: str = "",
    focus_time: str = "",
    image_path: str | None = None,
    reference_images: list[dict] | None = None,
    model_family: str = "anima",
) -> dict:
    """Call configured maker LLM route to suggest ONLY the selected place's tags and natural_language."""
    if not focus_place or focus_place not in (before.get("places") or {}):
        raise ValueError("먼저 편집할 장소를 선택하거나 추가하세요.")

    vision_images: list[tuple[str, ...]] = []
    manifest: list[dict] = []
    has_references = bool(reference_images)

    if image_path:
        if has_references:
            encoded = character_maker._encode_vision_image_with_banner(
                image_path,
                banner_text="GENERATED_CURRENT | Current generated background (edit target)",
                short_banner="GENERATED_CURRENT",
                banner_color="#1d4ed8",
            )
        else:
            encoded = character_maker._encode_vision_image(image_path)
        if encoded is None:
            raise ValueError("피드백할 이미지를 읽지 못했습니다. 다시 생성하세요.")
        if has_references:
            vision_images.append((
                encoded[0], encoded[1],
                "Image 1 [GENERATED_CURRENT] (blue top banner): Existing generated background image to inspect and refine.",
            ))
            manifest.append({
                "position": 1, "role": "CURRENT", "image_id": "GENERATED_CURRENT", "banner_color": "blue",
                "description": "Existing generated background image of the target place to inspect and refine.",
            })
        else:
            vision_images.append(encoded)
            manifest.append({
                "position": 1, "role": "CURRENT",
                "description": "Latest selected generated background image of the target place to refine.",
            })

    for ref_index, ref in enumerate(reference_images or [], start=1):
        ref_path = ref.get("path", "") if isinstance(ref, dict) else str(ref or "")
        ref_name = str(ref.get("name") or "").strip() if isinstance(ref, dict) else os.path.basename(ref_path)
        image_id = f"USER_UPLOADED_{ref_index}"
        encoded = character_maker._encode_vision_image_with_banner(
            ref_path,
            banner_text=f"{image_id} | User-uploaded reference image",
            short_banner=image_id,
            banner_color="#be123c",
        )
        if encoded is None:
            raise ValueError("첨부한 참고 이미지를 읽지 못했습니다. 다시 업로드해 주세요.")
        position = len(vision_images) + 1
        label_name = f" ({ref_name})" if ref_name else ""
        vision_images.append((
            encoded[0], encoded[1],
            f"Image {position} [{image_id}] (red top banner): Reference image #{ref_index} uploaded by the user{label_name}.",
        ))
        manifest.append({
            "position": position, "role": "REF", "image_id": image_id,
            "reference_index": ref_index, "banner_color": "red", "name": ref_name,
            "description": f"User-uploaded reference scenery image #{ref_index} attached to this message.",
        })

    system_prompt = (
        _BG_MAKER_SYSTEM_PROMPT_MOSS
        if str(model_family).strip().lower() == "moss"
        else _BG_MAKER_SYSTEM_PROMPT_ANIMA
    )
    place_obj = before["places"][focus_place]
    context_payload = {
        "theme_name": theme_name,
        "place_name": focus_place,
        "current_place_tags": place_obj.get("tags", []),
        "current_place_natural_language": place_obj.get("natural_language", ""),
        **({"preview_time_slot_note": f"Preview image uses separate time-of-day slot '{focus_time}'; keep place output time-neutral."} if focus_time else {}),
        **({"images": manifest} if manifest else {}),
    }

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
    ]
    for entry in history[-12:]:
        role = entry.get("role")
        content = str(entry.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({
        "role": "user",
        "content": (
            "Current place state (JSON):\n"
            f"{json.dumps(context_payload, ensure_ascii=False, indent=2)}\n\n"
            f"User request:\n{message.strip()}"
        ),
    })

    images = vision_images or None
    task_key = "character_maker_feedback" if images else "character_maker_draft"
    call_label = "배경 에셋 메이커"
    t0 = time.perf_counter()
    sink: dict = {}
    last_raw_result: str | None = None

    def _validator(raw_text: str) -> tuple[bool, str]:
        parsed, reason = _parse_bg_maker_llm_payload(raw_text, before, focus_place, focus_time)
        return (parsed is not None, reason)

    def _on_attempt_fail(info: dict) -> None:
        nonlocal last_raw_result
        try:
            from modes.lighbd_service import _log_lighbd_history
            attempt_raw = info.get("result")
            if attempt_raw:
                last_raw_result = str(attempt_raw)
            snap = dict(sink or {})
            _log_lighbd_history({
                "ts": datetime.now().isoformat(timespec="seconds"),
                "prompt_id": task_key,
                "call_name": call_label,
                "input": messages,
                "output": str(info.get("result") or ""),
                "completion_tokens": int(snap.get("completion_tokens") or 0),
                "prompt_tokens": int(snap.get("prompt_tokens") or 0),
                "elapsed": 0.0,
                "tps": 0.0,
                "status": "error",
                "error": (
                    f"[재시도 {info.get('phase')} {info.get('slot')} "
                    f"{info.get('attempt')}/{info.get('total_attempts')}] {info.get('reason')}"
                ),
            })
        except Exception:
            traceback.print_exc()

    try:
        if images:
            raw = await llm_service.callLLMVisionTask(
                task_key,
                messages,
                images=images,
                json_mode=True,
                result_validator=_validator,
                metadata_sink=sink,
                on_attempt_failure=_on_attempt_fail,
            )
        else:
            raw = await llm_service.callLLMTask(
                task_key,
                messages,
                json_mode=True,
                result_validator=_validator,
                metadata_sink=sink,
                on_attempt_failure=_on_attempt_fail,
            )
    except Exception as exc:
        character_maker._log_cm_history(
            task_key, messages, f"[LLM 실패] {exc}", t0,
            status="error", error=str(exc), sink=sink, call_label=call_label,
        )
        raise

    if isinstance(raw, str) and raw.strip().startswith("[LLM 실패]"):
        character_maker._log_cm_history(
            task_key, messages, last_raw_result or raw, t0,
            status="error", error=raw, sink=sink, call_label=call_label,
        )
        raise ValueError(raw)

    parsed, reason = _parse_bg_maker_llm_payload(raw, before, focus_place, focus_time)
    if parsed is None:
        character_maker._log_cm_history(
            task_key, messages, raw, t0, status="error", error=reason, sink=sink, call_label=call_label,
        )
        raise ValueError(reason)

    character_maker._log_cm_history(
        task_key, messages, raw, t0, status="ok", sink=sink, call_label=call_label,
    )
    return parsed


class AssetBgMaker:
    """Per-theme persistent maker proposals and non-destructive version restores."""

    def __init__(self, themes_store, root):
        self.themes = themes_store
        self.root = Path(root)
        self.running = set()

    def path(self, name: str) -> Path:
        return self.root / f"{fingerprint(name)}.json"

    def uploads_dir(self, name: str) -> Path:
        return self.root / "uploads" / fingerprint(name)

    @staticmethod
    def _uploaded_images_by_id(state: dict) -> dict:
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

    def _remove_uploaded_files(self, name: str, items: list) -> None:
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

    def _clear_uploads_dir(self, name: str) -> None:
        uploads_root = self.uploads_dir(name)
        if uploads_root.is_dir():
            shutil.rmtree(uploads_root, ignore_errors=True)

    @staticmethod
    def _public_uploads(name: str, items: list) -> list[dict]:
        public = []
        for item in items or []:
            if not isinstance(item, dict) or not item.get("id"):
                continue
            public.append({
                "id": item["id"],
                "name": str(item.get("name") or ""),
                "url": "/api/bg-asset/theme-maker/image?" + urlencode({"name": name, "upload_id": item["id"]}),
            })
        return public

    def _theme(self, name: str):
        if not isinstance(name, str) or not name:
            raise ValueError("테마를 선택하세요.")
        catalog = self.themes.catalog()
        if name not in catalog["themes"]:
            raise ValueError("테마를 찾을 수 없습니다. 다시 선택하세요.")
        return catalog, catalog["themes"][name]

    def _read(self, name: str) -> dict:
        path = self.path(name)
        state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
            "name": name, "versions": [], "active_version_id": "", "pending": None, "error": "",
        }
        if state.get("name") != name or not isinstance(state.get("versions"), list):
            raise ValueError("배경 테마 버전 파일을 확인하세요.")
        if state.get("pending") and state["pending"]["id"] not in self.running:
            orphan_uploads = list(state["pending"].get("uploaded_images") or [])
            state.update(pending=None, error="서버 재시작으로 AI 요청이 중단되었습니다. 다시 요청하세요.")
            write_json_atomic(path, state)
            self._remove_uploaded_files(name, orphan_uploads)
        return state

    def _public(self, state: dict, theme: dict) -> dict:
        result = {**copy.deepcopy(state), "revision": fingerprint(state), "theme_revision": fingerprint(theme)}
        for version in result["versions"]:
            path = version.pop("image_path", "")
            if path:
                version["image_url"] = "/api/bg-asset/theme-maker/image?" + urlencode({
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

    def image_path(self, name: str, version_id: str) -> str:
        with settings_lock(self.themes.path):
            self._theme(name)
            version = next((v for v in self._read(name)["versions"] if v["id"] == version_id), None)
            path = version.get("image_path", "") if version else ""
            if not path or not Path(path).is_file():
                raise ValueError("이 버전의 생성 이미지를 찾을 수 없습니다. 다시 생성하세요.")
            return path

    def uploaded_image_entries(self, name: str, items: list) -> list[dict]:
        with settings_lock(self.themes.path):
            self._theme(name)
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

    def uploaded_image_path(self, name: str, upload_id: str) -> str:
        entries = self.uploaded_image_entries(name, [{"id": upload_id}])
        return entries[0]["path"]

    def read(self, name: str) -> dict:
        with settings_lock(self.themes.path):
            _, theme = self._theme(name)
            return self._public(self._read(name), theme)

    def _checked(self, body: dict):
        name = body.get("name")
        catalog, theme = self._theme(name)
        state = self._read(name)
        if body.get("revision") != fingerprint(state) or body.get("theme_revision") != fingerprint(theme):
            raise ValueError("테마 또는 버전이 변경되었습니다. 새로고침 후 다시 시도하세요.")
        if state.get("pending"):
            raise ValueError("AI 결과를 기다린 뒤 다시 시도하세요.")
        theme_data = _validate_theme_payload(body.get("theme_data"))
        if theme_data != normalize_bg_theme(theme, strict=False):
            raise ValueError("화면과 저장된 배경 설정이 다릅니다. 저장이 끝난 뒤 다시 시도하세요.")
        return name, catalog, theme, state, theme_data

    @staticmethod
    def _append(state: dict, kind: str, before: dict, after: dict, **extra) -> dict:
        version = {
            "id": uuid.uuid4().hex, "number": len(state["versions"]) + 1, "kind": kind,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "before": copy.deepcopy(before), "after": copy.deepcopy(after), **extra,
        }
        state["versions"].append(version)
        return version

    def _checkpoint(self, state: dict, theme_data: dict, focus_place: str = "", focus_time: str = "") -> dict:
        active = next((v for v in state["versions"] if v["id"] == state["active_version_id"]), None)
        if active and active["after"] == theme_data:
            return active
        fp, ft = _pick_focus(theme_data, focus_place, focus_time)
        version = self._append(
            state, "snapshot", theme_data, theme_data,
            message="현재 장소 설정 보관", focus_place=fp, focus_time=ft,
        )
        state["active_version_id"] = version["id"]
        return version

    def change(self, body: dict) -> dict:
        with settings_lock(self.themes.path):
            name, catalog, theme, state, theme_data = self._checked(body)
            action = body.get("action")
            focus_place = str(body.get("focus_place") or "").strip()
            focus_time = str(body.get("focus_time") or "").strip()
            if action == "snapshot":
                self._checkpoint(state, theme_data, focus_place, focus_time)
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
                    self.themes.lock.require(body)
                # Apply ONLY the focused place's base tags and natural_language onto current theme data,
                # preserving all current time slots, theme common/negative tags, and other places.
                target_place = str(target.get("focus_place") or focus_place or "").strip()
                target_after = _validate_theme_payload(target["after"])
                if target_place and target_place in target_after["places"] and target_place in theme_data["places"]:
                    after = copy.deepcopy(theme_data)
                    after["places"][target_place]["tags"] = copy.deepcopy(target_after["places"][target_place]["tags"])
                    after["places"][target_place]["natural_language"] = target_after["places"][target_place]["natural_language"]
                    after = _validate_theme_payload(after)
                else:
                    after = target_after
                self._checkpoint(state, theme_data, target_place or focus_place, focus_time)
                fp, ft = _pick_focus(after, target_place or target.get("focus_place", ""), focus_time or target.get("focus_time", ""))
                version = self._append(
                    state, action, theme_data, after, source_id=target["id"],
                    message=f"v{target['number']} {'적용' if action == 'apply' else '롤백'}",
                    focus_place=fp, focus_time=ft,
                    **{key: target[key] for key in ("image_path", "image_updated_at") if key in target},
                )
                state.update(active_version_id=version["id"], error="")
                path = self.themes.builtin_file if scope == "builtin" else self.themes.path
                document = self.themes._read_file(path)
                if scope == "builtin":
                    document["version"] = 1
                document["themes"][name] = after
                write_scope_documents({self.path(name): state, path: document})
                theme = after
            else:
                raise ValueError("지원하지 않는 메이커 요청입니다.")
            return self._public(state, theme)

    def begin(self, body: dict, uploaded_files=None):
        validated_uploads = validate_uploaded_images(uploaded_files)
        with settings_lock(self.themes.path):
            name, _, theme, state, theme_data = self._checked(body)
            generating = body.get("action") == "generate"
            if generating and validated_uploads:
                raise ValueError("이미지는 AI 대화 요청에만 첨부할 수 있습니다.")
            message = "이미지 생성" if generating else body.get("message")
            if not isinstance(message, str) or not message.strip() or len(message) > 8000:
                raise ValueError("AI 요청을 1~8,000자로 입력하세요.")
            focus_place = str(body.get("focus_place") or "").strip()
            focus_time = str(body.get("focus_time") or "").strip()
            base_id = body.get("base_id")
            if base_id:
                base = next((v for v in state["versions"] if v["id"] == base_id), None)
                if base is None:
                    raise ValueError("대화 기준 버전을 찾을 수 없습니다.")
            else:
                base = self._checkpoint(state, theme_data, focus_place, focus_time)

            base_theme = normalize_bg_theme(base["after"], strict=False)
            if not generating and focus_place and focus_place in theme_data["places"]:
                merged = copy.deepcopy(theme_data)
                if focus_place in base_theme["places"] and (not base.get("focus_place") or base.get("focus_place") == focus_place):
                    merged["places"][focus_place]["tags"] = copy.deepcopy(base_theme["places"][focus_place]["tags"])
                    merged["places"][focus_place]["natural_language"] = base_theme["places"][focus_place]["natural_language"]
                base_theme = merged
            eff_place, eff_time = _pick_focus(
                base_theme,
                focus_place or str(base.get("focus_place") or ""),
                focus_time or str(base.get("focus_time") or ""),
            )
            if not eff_place:
                raise ValueError("먼저 편집할 장소를 추가하고 선택하세요.")

            ancestry, cursor, seen = [], base, set()
            by_id = {v["id"]: v for v in state["versions"]}
            while cursor and cursor["id"] not in seen and len(ancestry) < 6:
                seen.add(cursor["id"])
                if cursor["kind"] == "proposal" and cursor.get("focus_place") == eff_place:
                    ancestry.append(cursor)
                cursor = by_id.get(cursor.get("parent_id") or cursor.get("source_id"))
            history = [entry for version in reversed(ancestry) for entry in (
                {"role": "user", "content": version["message"]},
                {"role": "assistant", "content": version.get("assistant_message", "")},
            )]
            pending = {
                "id": uuid.uuid4().hex, "message": message.strip(), "base_id": base["id"],
                "phase": "generate" if generating else "propose",
                "focus_place": eff_place, "focus_time": eff_time,
            }
            if generating:
                pending["version_id"] = base["id"]
                base["focus_place"] = eff_place
                base["focus_time"] = eff_time
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
            return self._public(state, theme), copy.deepcopy(base_theme), history, eff_place, eff_time

    def finish(self, name: str, request_id: str, before: dict, result=None, error: str = "", *, generate: bool = False):
        with settings_lock(self.themes.path):
            state = self._read(name)
            pending = state.get("pending")
            if not pending or pending["id"] != request_id:
                return None
            orphan_uploads = []
            version = None
            will_generate = False
            if result is not None:
                after = _validate_theme_payload(result["after"])
                focus_place, focus_time = _pick_focus(after, result.get("focus_place", ""), result.get("focus_time", ""))
                uploaded_images = copy.deepcopy(pending.pop("uploaded_images", None) or [])
                extra = {
                    "feedback_image_version_id": pending.get("image_version_id", ""),
                    "focus_place": focus_place,
                    "focus_time": focus_time,
                }
                if uploaded_images:
                    extra["uploaded_images"] = uploaded_images
                version = self._append(
                    state, "proposal", before, after, parent_id=pending["base_id"],
                    message=pending["message"], assistant_message=str(result.get("assistant_message", "")),
                    **extra,
                )
                will_generate = bool(generate and focus_place)
                if will_generate:
                    pending.update(phase="generate", version_id=version["id"], focus_place=focus_place, focus_time=focus_time)
            else:
                orphan_uploads = list(pending.get("uploaded_images") or [])
            if error and pending.get("version_id"):
                target = next((v for v in state["versions"] if v["id"] == pending["version_id"]), None)
                if target:
                    target["image_error"] = error
            state.update(pending=pending if will_generate and not error else None, error=error)
            write_json_atomic(self.path(name), state)
            if orphan_uploads:
                self._remove_uploaded_files(name, orphan_uploads)
            if not state["pending"]:
                self.running.discard(request_id)
            return copy.deepcopy(version) if version is not None else None

    def finish_image(self, name: str, request_id: str, result: dict) -> None:
        with settings_lock(self.themes.path):
            state = self._read(name)
            pending = state.get("pending")
            if not pending or pending["id"] != request_id:
                return
            path = result.get("local_path") if isinstance(result, dict) and result.get("success") else None
            if not path or not Path(path).is_file():
                raise ValueError((result.get("error") if isinstance(result, dict) else "") or "생성 이미지를 저장하지 못했습니다.")
            version = next(v for v in state["versions"] if v["id"] == pending["version_id"])
            version.update(
                image_path=str(path),
                image_updated_at=datetime.now(timezone.utc).isoformat(),
                image_error="",
                focus_place=pending.get("focus_place", version.get("focus_place", "")),
                focus_time=pending.get("focus_time", version.get("focus_time", "")),
            )
            state.update(pending=None, error="")
            write_json_atomic(self.path(name), state)
            self.running.discard(request_id)


def bg_generation_job(template: dict, theme_data: dict, place_name: str, time_name: str) -> dict:
    """Inject resolved theme + place + time tags/NL/LoRA into a prepared background job template."""
    from illustration_json import is_moss_document

    job = copy.deepcopy(template)
    place = (theme_data.get("places") or {}).get(place_name) or {}
    time_entry = (place.get("times") or {}).get(time_name) or {}
    positive_tags = [
        tag.strip()
        for tag in [
            *(theme_data.get("common_tags") or []),
            *(place.get("tags") or []),
            *(time_entry.get("tags") or []),
        ]
        if isinstance(tag, str) and tag.strip()
    ]
    negative_tags = [
        tag.strip()
        for tag in (theme_data.get("negative_tags") or [])
        if isinstance(tag, str) and tag.strip()
    ]
    natural_language = "\n".join(
        part.strip()
        for part in (place.get("natural_language", ""), time_entry.get("natural_language", ""))
        if isinstance(part, str) and part.strip()
    )

    job["place"] = place_name or "기본"
    job["outfit"] = job["place"]
    job["time"] = time_name or "기본"
    job["atmosphere"] = job["time"]
    job["expression"] = job["time"]

    if is_moss_document(job.get("prompt_json")):
        from modes.moss_prompt_builder import MossPromptBuilder

        moss = job["prompt_json"]["moss"]
        summary = moss.setdefault("summary", {})
        parts = [summary.get("caption", ""), *positive_tags, natural_language]
        summary["caption"] = ", ".join(part.strip() for part in parts if isinstance(part, str) and part.strip())
        if negative_tags:
            workflow = moss.setdefault("workflow", {})
            existing_neg = str(workflow.get("negative") or "").strip()
            neg_parts = [existing_neg, *negative_tags]
            workflow["negative"] = ", ".join(part for part in neg_parts if part)
        job["positive_prompt"] = MossPromptBuilder.dumps(job["prompt_json"])
        return job

    fields = job["prompt_json"]["fields"]
    for field in ("ANIMA_CONTENT", "SDXL_CONTENT"):
        if field not in fields:
            continue
        parts = [fields[field], *positive_tags]
        if field == "ANIMA_CONTENT" and natural_language:
            parts.append(natural_language)
        fields[field] = ", ".join(part.strip() for part in parts if isinstance(part, str) and part.strip())
    if negative_tags:
        for neg_field in ("ANIMA_NEGATIVE", "SDXL_NEGATIVE"):
            if neg_field in fields:
                existing_neg = str(fields[neg_field] or "").strip()
                fields[neg_field] = ", ".join(part for part in (existing_neg, *negative_tags) if part)
    job["positive_prompt"] = json.dumps(job["prompt_json"], ensure_ascii=False, allow_nan=False)
    return job


def register_bg_theme_maker_runtime(server, prepare_generation=None):
    store = AssetBgMaker(
        server.bg_asset_mode.theme_presets,
        Path(server.CONFIG_FILE).parent / "bg_asset_theme_versions",
    )

    async def read_request(request: web.Request):
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
            return json.loads(payload_raw), uploaded_files
        return await request.json(), []

    async def endpoint(request: web.Request) -> web.Response:
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
                        template["theme"] = body.get("name")
                        template["character"] = body.get("name")
                        snapshot = prepare_generation(template)
                        template.update(
                            storage_group="character_maker",
                            storage_session=f"asset-{fingerprint('bg:' + str(body.get('name') or ''))}",
                        )
                    elif body.get("action") == "generate":
                        raise ValueError("이미지 생성 설정을 확인하세요.")

                    result, before, history, eff_place, eff_time = store.begin(body, uploaded_files=uploaded_files)
                    name, pending = result["name"], result["pending"]

                    async def enqueue_image(theme_values: dict, place_name: str, time_name: str):
                        job_payload = bg_generation_job(template, theme_values, place_name, time_name)
                        item = await server.queue_manager.add_item(
                            "asset_generation",
                            label=f"배경 메이커 이미지 · {name}/{place_name}/{time_name or '기본'}",
                            params={
                                "body": job_payload,
                                "asset_model_preset": snapshot,
                                "asset_workspace": "bg_asset",
                            },
                        )

                        def image_completed(future):
                            try:
                                store.finish_image(name, pending["id"], future.result())
                            except BaseException as exc:
                                store.finish(
                                    name, pending["id"], before,
                                    error=str(exc) or "이미지 생성이 취소되었습니다. 다시 생성하세요.",
                                )

                        item.completion_future.add_done_callback(image_completed)

                    async def run(_item):
                        image_path = store.image_path(name, pending["image_version_id"]) if pending.get("image_version_id") else None
                        kwargs = {
                            "focus_place": eff_place,
                            "focus_time": eff_time,
                            "image_path": image_path,
                        }
                        if pending.get("uploaded_images"):
                            kwargs["reference_images"] = store.uploaded_image_entries(name, pending["uploaded_images"])
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
                        if str(workflow_type).strip().lower() == "moss":
                            kwargs["model_family"] = "moss"
                        proposal = await suggest_bg_theme_values(
                            server.character_maker,
                            name, before, pending["message"], history, **kwargs,
                        )
                        version = store.finish(name, pending["id"], before, proposal, generate=template is not None)
                        if template is not None and version and version.get("focus_place"):
                            await enqueue_image(version["after"], version["focus_place"], version.get("focus_time", ""))
                        return {"success": True}

                    def completed(future):
                        try:
                            future.result()
                        except BaseException as exc:
                            store.finish(
                                name, pending["id"], before,
                                error=str(exc) or "AI 요청이 취소되었습니다. 다시 요청하세요.",
                            )

                    try:
                        if body.get("action") == "generate":
                            await enqueue_image(before, eff_place, eff_time)
                        else:
                            item = await server.queue_manager.add_item(
                                "character_maker",
                                label=f"배경 메이커 · {name}/{eff_place}",
                                params={"theme": name, "place": eff_place, "asset_workspace": "bg_asset"},
                                runtime_handler=run,
                            )
                            item.completion_future.add_done_callback(completed)
                    except Exception as exc:
                        store.finish(name, pending["id"], before, error=str(exc))
                        raise
                else:
                    result = store.change(body)
                    if body.get("action") in {"apply", "rollback"}:
                        server.bg_asset_mode.load_tags()
            return web.json_response(result, headers={"Cache-Control": "no-store"})
        except (ValueError, TypeError, OSError, KeyError) as exc:
            return web.json_response({"error": str(exc)}, status=400)

    async def image(request: web.Request) -> web.StreamResponse:
        try:
            name = request.query.get("name", "")
            upload_id = request.query.get("upload_id", "")
            path = (
                store.uploaded_image_path(name, upload_id)
                if upload_id
                else store.image_path(name, request.query.get("version_id", ""))
            )
            return web.FileResponse(path, headers={"Cache-Control": "no-store"})
        except (ValueError, OSError):
            raise web.HTTPNotFound()

    server.app.router.add_get("/api/bg-asset/theme-maker", endpoint)
    server.app.router.add_post("/api/bg-asset/theme-maker", endpoint)
    server.app.router.add_get("/api/bg-asset/theme-maker/image", image)
    return store
