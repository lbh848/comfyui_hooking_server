"""Regenerate a generated bot image, then refresh its normal reference assets."""
from __future__ import annotations

import asyncio
from copy import deepcopy
import glob
from importlib import import_module
import io
import json
import os
from pathlib import Path
import re
import time
import traceback
import uuid

from aiohttp import web
from PIL import Image, PngImagePlugin

from modes import illustration_context_pipeline as pipeline
from modes import illustration_scene_pipeline as scene_pipeline
from modes.illustration_npc import _character_dir, character_image_descriptor, is_generated_image, save_tracked_face
from modes.visual_profiles import effective_character_cards, store_visual_cards, tag_values
from settings_illustration_prepare import _call
import workflow_profiles

storage = import_module("modes.bot_mode")

# In-flight emotion workshop jobs, shared with the 감정 삽화 캐릭터 panel header.
_EMOTION_ACTIVITY_STORES: dict[int, list[dict]] = {}


def _track_emotion_activity(server, runs):
    """Register one emotion job store so the activity endpoint can count it."""
    stores = _EMOTION_ACTIVITY_STORES.setdefault(id(server), [])
    if runs not in stores:
        stores.append(runs)


def _emotion_activity(server, bot_name):
    active = [
        run
        for runs in _EMOTION_ACTIVITY_STORES.get(id(server), [])
        for run in runs.values()
        if run.get("status") in ("generating", "preparing")
        and str(run.get("bot_name") or "") == bot_name
    ]
    return {
        "count": len(active),
        "characters": sorted({str(run.get("char_name") or "") for run in active} - {""}),
    }


def register_character_image_regeneration(server):
    runs = {}
    tasks = {}

    def target(bot_name, char_name, filename, visual_card_id):
        bot = next((entry for entry in storage._load_bot_data().get("bots", []) if entry.get("name") == bot_name), None)
        character = next((entry for entry in (bot or {}).get("characters", []) if entry.get("name") == char_name), None)
        if character is None:
            raise ValueError("캐릭터를 찾을 수 없습니다.")
        directory = _character_dir(storage, bot_name, char_name)
        image_path = (directory / filename).resolve()
        if not filename or filename != Path(filename).name or image_path.parent != directory or not image_path.is_file():
            raise ValueError("재생성할 이미지를 찾을 수 없습니다.")
        prompt_path = image_path.with_name(image_path.stem + "_prompt.json")
        metadata = json.loads(prompt_path.read_text(encoding="utf-8")) if prompt_path.is_file() else {}
        if not isinstance(metadata, dict) or not is_generated_image(filename, metadata, npc_generated=character.get("npc_generated") is True):
            raise ValueError("삽화 봇에서 생성한 이미지만 재생성할 수 있습니다.")
        cards, _ = effective_character_cards(character, None)
        card = next((entry for entry in cards if entry["id"] == visual_card_id), None)
        if card is None:
            raise ValueError("재생성에 사용할 캐릭터 카드를 저장해 주세요.")
        render_root = deepcopy(character)
        store_visual_cards(render_root, [card])
        # Rebuild identity from the edited card, without the old image's cache.
        render_root["_npc_pending"] = True
        render_bot = deepcopy(bot)
        render_bot["characters"] = [render_root]
        return image_path, prompt_path, metadata, render_root, render_bot

    async def wait_item(run, item):
        run["item_id"] = item.id
        try:
            result = await asyncio.shield(item.completion_future)
            if item.status != "completed" or isinstance(result, dict) and (result.get("error") or result.get("success") is False):
                raise RuntimeError(item.error or "작업을 완료하지 못했습니다.")
            return result
        except asyncio.CancelledError:
            if not item.completion_future.done():
                await server.queue_manager.cancel_item(item.id)
            raise
        except Exception:
            if item.status == "cancelled":
                raise asyncio.CancelledError()
            raise

    async def regenerate(run, prepared, snapshot):
        image_path, prompt_path, metadata, root, render_bot = prepared
        prompt_id = f"character-regenerate-{uuid.uuid4().hex}"
        temporary = image_path.with_name(f".{prompt_id}.tmp")
        try:
            original_version = image_path.stat().st_mtime_ns
            toggles = {**snapshot["illustration_context_toggles"], "positive_note": "", "negative_note": ""}
            # 감정 삽화의 NPC 대표 이미지를 다시 만들 때도 프리셋 설정의
            # 복장 대표 표정 태그를 유지해 자유 포즈로 돌아가지 않게 한다.
            emotion_tags = (
                pipeline.npc_representative_emotion_tags(
                    toggles, str(snapshot.get("workshop") or "illustration"))
                if root.get("npc_generated") else [])
            descriptor = character_image_descriptor(root, emotion_tags=emotion_tags)
            positive, negative = pipeline.build_raw_prompt(
                descriptor, "", pipeline.load_prompt_files(snapshot.get("workshop")), toggles,
            )
            provider = snapshot["provider"]
            workflow_type = snapshot["illustration_workflow_type"]
            prompt_format = workflow_profiles.illustration_prompt_format(workflow_type, "comfy" if provider == "hybrid" else provider)
            prompt_data = {
                "positive": {"class_type": "STRING", "_meta": {"title": "긍정프롬프트"}, "inputs": {"value": positive}},
                "negative": {"class_type": "STRING", "_meta": {"title": "부정프롬프트"}, "inputs": {"value": negative}},
            }
            snapshot.update(generation_mode="character_auto", npc_render_bot=render_bot)
            server.prompts[prompt_id] = {
                "status": "running", "prompt": prompt_data, "client_id": "", "extra_data": {},
                "outputs": {}, "filename": None, "save_node_id": None, "image_bytes": None,
                "timestamp": time.time(), "_illustration_runtime_snapshot": snapshot,
            }
            # MOSS uses the NPC card values as a summary-only layout, so the
            # representative needs no regional bbox/tracking path.
            moss_npc = bool(root.get("npc_generated")
                            and toggles.get("illustration_model_family") == "moss")
            params = {"prompt_id": prompt_id, "prompt_data": prompt_data, "provider": provider,
                      # 큐 실행 시점에도 감정 삽화 봇 저장소/그리기 설정을 쓰도록
                      # 작업에 워크숍 식별자를 함께 보관한다.
                      "workshop": str(snapshot.get("workshop") or "illustration"),
                      "illustration_flow_skip": True,
                      "raw_body": {"illustration_provider": provider, "illustration_prompt_format": prompt_format,
                                   "illustration_defer_postprocess": True,
                                   **({"illustration_npc_representative": True}
                                      if root.get("npc_generated") else {})}}
            if root.get("npc_generated") and not moss_npc:
                params["npc_tracking_root"] = deepcopy(root)
            if provider == "hybrid":
                params["hybrid_prompt_formats"] = {value: workflow_profiles.illustration_prompt_format(workflow_type, value)
                                                    for value in ("comfy", "chansub")}
            item = await server.queue_manager.add_item("illustration", f"캐릭터 이미지 재생성 · {run['char_name']}", params, priority=0)
            await wait_item(run, item)
            entry = server.prompts.get(prompt_id, {})
            pixels = entry.get("_deferred_image_bytes")
            if not pixels:
                raise RuntimeError(entry.get("error") or "재생성한 이미지가 없습니다.")
            prompt = descriptor["characters"][0]["positive"]
            png_info = PngImagePlugin.PngInfo()
            png_info.add_text("prompt", prompt)
            with Image.open(io.BytesIO(pixels)) as image:
                image.save(temporary, format="PNG", pnginfo=png_info)
            async with server.bot_mode._lock:
                # A removed/replaced image must not reappear after the GPU finishes.
                target(run["bot_name"], run["char_name"], run["filename"], run["visual_card_id"])
                if image_path.stat().st_mtime_ns != original_version:
                    raise ValueError("생성 중 원본 이미지가 변경되었습니다. 새 이미지를 확인한 뒤 다시 실행하세요.")
                storage.write_json_atomic(prompt_path, {**metadata, "prompt": prompt, "negative": negative,
                                                       "source": "character_auto"})
                try:
                    os.replace(temporary, image_path)
                except BaseException:
                    storage.write_json_atomic(prompt_path, metadata)
                    raise
                run.update(image_replaced=True, status="preparing", message="대표 이미지 참조 준비 중…")
                result = await _call(server.data_patcher.handle_data_patch,
                                     {"bot_name": run["bot_name"], "char_name": run["char_name"]})
                if root.get("npc_generated"):
                    info = (entry.get("_deferred_finalize") or {}).get("conversion_info") or {}
                    save_tracked_face(
                        directory=storage.bot_visual_artifact_dir(run["bot_name"], run["char_name"], run["visual_card_id"]),
                        source_directory=image_path.parent, source_filename=image_path.name,
                        character_name=run["char_name"], face_positions=info.get("face_positions") or [],
                        settings=storage._load_patch_settings(run["bot_name"]),
                    )
                    data = storage._load_bot_data()
                    bot = next(value for value in data["bots"] if value["name"] == run["bot_name"])
                    character = next(value for value in bot["characters"] if value["name"] == run["char_name"])
                    if character.get("npc_face_source") != "representative_mask_tracking":
                        character["npc_face_source"] = "representative_mask_tracking"
                        storage._save_bot_data(data)
                    run.update(status="completed", message="대표 이미지와 생성 시 추적된 FACE 처리를 완료했습니다.")
                    return
            from comfy_installer.spatial_face_workflow import uses_spatial_faces
            from illustration_model_presets import read_workflow
            profile = workflow_profiles.illustration_local_profile(workflow_type)
            source = (snapshot.get("illustration_model_preset") or {}).get("config", {}).get("illustration_workflow_source_paths", {}).get(profile)
            workflow, _ = read_workflow(source or server.get_comfy_workflow_source_path(workflow_type))
            if uses_spatial_faces(workflow):
                run.update(status="completed", message="이미지 교체를 완료했습니다.")
                return
            # The normal patch stages the whole character, so rebuild every card's
            # reference cache before declaring completion.
            targets = result.get("visual_targets") or []
            if not targets:
                raise ValueError("FACE-IPAdapter를 실행하려면 캐릭터 카드에 대표 이미지를 지정하세요.")
            for visual_target in targets:
                item = await server.queue_manager.add_item("data_patch_utility", f"FACE-IPAdapter · {run['char_name']}", {
                    "bot_name": run["bot_name"], "char_name": run["char_name"],
                    **{key: visual_target[key] for key in ("visual_card_id", "visual_card_label", "visual_card_index")},
                }, priority=0)
                await wait_item(run, item)
            run.update(status="completed", message="이미지 교체와 FACE-IPAdapter 처리를 완료했습니다.")
        except asyncio.CancelledError:
            run.update(status="cancelled", message="", error="이미지는 교체되었지만 FACE-IPAdapter 처리가 중단되었습니다." if run["image_replaced"] else "재생성이 중단되었습니다.")
        except Exception as exc:
            traceback.print_exc()
            run.update(status="failed", message="", error=("이미지는 교체되었지만 FACE-IPAdapter 처리에 실패했습니다: " if run["image_replaced"] else "재생성에 실패했습니다: ") + str(exc))
        finally:
            tasks.pop((run["bot_name"], run["char_name"]), None)
            temporary.unlink(missing_ok=True)
            server._discard_deferred_illustration_prompt(prompt_id, "캐릭터 이미지 재생성 종료")

    async def handle(request):
        if request.method == "GET":
            key = (request.query.get("bot_name", ""), request.query.get("char_name", ""))
            return web.json_response({"run": runs.get(key)})
        try:
            body = await request.json()
            if not isinstance(body, dict) or any(not isinstance(body.get(key), str) or not body[key] for key in ("bot_name", "char_name", "filename", "visual_card_id")):
                raise ValueError("재생성할 캐릭터와 이미지를 선택하세요.")
            key = (body["bot_name"], body["char_name"])
            if key in tasks:
                return web.json_response({"run": runs[key]}, status=202)
            prepared = target(*key, body["filename"], body["visual_card_id"])
            # The request's bot scope decides which workshop's bot this is.
            emotion_scope = storage.bot_scope() == "emotion"
            snapshot = server._capture_illustration_runtime_snapshot(
                {
                    **server.app_config,
                    "emotion_bot_selected" if emotion_scope else "bot_selected": body["bot_name"],
                },
                **({"workshop": "emotion"} if emotion_scope else {}),
            )
            run = {key: body[key] for key in ("bot_name", "char_name", "filename", "visual_card_id")}
            run.update(id=uuid.uuid4().hex, status="generating", message="캐릭터 이미지 재생성 중…", error="", image_replaced=False, item_id="")
            runs[key] = run
            tasks[key] = asyncio.create_task(regenerate(run, prepared, snapshot))
            return web.json_response({"run": run}, status=202)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            return web.json_response({"error": str(exc)}, status=400)

    async def cleanup(_app):
        pending = list(tasks.values())
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

    server.app.router.add_get("/api/bot_mode/regenerate_image", handle)
    server.app.router.add_post("/api/bot_mode/regenerate_image", handle)
    server.app.on_cleanup.append(cleanup)


def _replace_outfit_tags(text: str, old_tags, new_tags) -> str:
    """Swap the outfit tag block in a stored prompt, leaving the rest untouched.

    The emotion asset backup keeps the exact prompt that produced the outfit cover.
    Replacing only the registered outfit tags therefore regenerates the same scene
    with a different outfit state instead of rebuilding the whole character prompt.
    """
    if not isinstance(text, str) or not text.strip():
        return text
    old = [str(tag).strip() for tag in (old_tags or []) if str(tag).strip()]
    new = [str(tag).strip() for tag in (new_tags or []) if str(tag).strip()]
    if not old or old == new:
        return text
    alternatives = "|".join(re.escape(tag) for tag in sorted(old, key=len, reverse=True))
    tag_pattern = rf"(?<![\w-])(?:{alternatives})(?![\w-])"
    # The stored prompt repeats the outfit block in several sections (CONTENT,
    # ALL, SDXL, ...). Replace each contiguous run so every section stays aligned.
    run_pattern = re.compile(
        rf"(?<![\w-])(?:{alternatives})(?:[ \t]*,[ \t]*(?:{alternatives}))*(?![\w-])"
    )
    updated = run_pattern.sub(", ".join(new), text)
    if updated == text:
        return text
    # Drop any old tag that sat outside a contiguous outfit run.
    updated = re.sub(tag_pattern, "", updated)
    # Collapse comma slots the removals left empty, keeping newlines intact.
    for _ in range(4):
        cleaned = re.sub(r"[ \t]*,[ \t]*(?=,)", ",", updated)
        cleaned = re.sub(r",[ \t]*,", ",", cleaned)
        cleaned = re.sub(r",[ \t]*(?=\n)", "", cleaned)
        cleaned = re.sub(r"\n[ \t]*,", "\n", cleaned)
        if cleaned == updated:
            break
        updated = cleaned
    return updated.strip()


def _image_extension(data: bytes) -> str:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    if data[:2] == b"\xff\xd8":
        return ".jpg"
    return ""


def _write_emotion_image(path, data):
    extension = os.path.splitext(path)[1].lower()
    temporary = path + ".tmp"
    source_extension = _image_extension(data)
    if source_extension and source_extension == extension:
        with open(temporary, "wb") as handle:
            handle.write(data)
    else:
        format_name = {".png": "PNG", ".webp": "WEBP", ".jpg": "JPEG", ".jpeg": "JPEG"}.get(extension)
        if not format_name:
            raise ValueError("지원하지 않는 감정 이미지 형식입니다.")
        with Image.open(io.BytesIO(data)) as image:
            image.save(temporary, format=format_name)
    os.replace(temporary, path)


def _server_backup_dirs(server):
    """워크숍별로 분리된 백업 폴더 목록(현재 컨텍스트 우선).

    삽화/감정 삽화는 백업 폴더가 다르므로 이름으로 찾을 때는 양쪽을 모두
    확인해야 한다. 구버전 server 모듈에는 헬퍼가 없을 수 있어 폴백을 둔다.
    """
    dirs = getattr(server, "backup_search_dirs", None)
    if callable(dirs):
        try:
            found = [str(path) for path in dirs() if path]
            if found:
                return found
        except Exception:
            pass
    return [str(server.WORKFLOW_BACKUP_DIR)]


def _server_backup_dir(server, backup_name):
    """백업 파일이 실제로 있는 폴더. 없으면 현재 컨텍스트의 폴더(쓰기 위치)."""
    finder = getattr(server, "find_backup_dir", None)
    if callable(finder):
        try:
            return str(finder(backup_name))
        except Exception:
            pass
    return str(server.WORKFLOW_BACKUP_DIR)


def _find_backup_file(server, backup_name, suffix):
    """이름 기반 백업 파일을 두 워크숍 폴더에서 찾는다."""
    finder = getattr(server, "find_backup_file", None)
    if callable(finder):
        try:
            path = str(finder(backup_name, suffix) or "")
            if path:
                return path
        except Exception:
            pass
    for directory in _server_backup_dirs(server):
        path = os.path.join(directory, f"{backup_name}{suffix}")
        if os.path.isfile(path):
            return path
    return ""


def _find_emotion_asset_backup(server, image_path):
    """이미지를 생성한 백업(프롬프트 기록)의 이름을 찾는다.

    A bulk overwrite can leave several backups pointing at the same store file;
    the newest record describes the image that is actually there now.
    """
    target = os.path.normcase(os.path.abspath(image_path))
    target_name = os.path.basename(target)
    target_dir = os.path.normcase(os.path.dirname(target))
    matches = []
    info_paths = []
    for directory in _server_backup_dirs(server):
        info_paths.extend(glob.glob(os.path.join(directory, "*_info.json")))
    for info_path in info_paths:
        try:
            with open(info_path, "r", encoding="utf-8") as handle:
                info = json.load(handle)
        except (OSError, ValueError):
            continue
        asset = info.get("emotion_asset") if isinstance(info, dict) else None
        if not isinstance(asset, dict):
            continue
        saved = str(asset.get("saved_path") or "").strip()
        name = os.path.basename(info_path)[: -len("_info.json")]
        if saved and os.path.normcase(os.path.abspath(saved)) == target:
            matches.append(name)
        # Older backups may not record an absolute path; match the folder too.
        elif saved and os.path.basename(saved) == target_name:
            asset_dir = os.path.normcase(os.path.abspath(str(asset.get("target_dir") or "")))
            if asset_dir and asset_dir == target_dir:
                matches.append(name)
    return max(matches) if matches else ""


def _load_emotion_prompt(server, backup_name):
    """감정 이미지의 원본 프롬프트, 재생성 결과를 보관할 사이드카, 그 복장 태그를 읽는다."""
    # 감정 삽화 백업은 감정 전용 폴더에 저장되므로 양쪽 폴더를 이름으로 찾는다.
    prompt_path = (_find_backup_file(server, backup_name, ".json")
                   or _find_backup_file(server, backup_name, ".txt"))
    sidecar_path = _find_backup_file(server, backup_name, "_outfit_prompt.json")
    if not sidecar_path:
        # 아직 사이드카가 없으면 백업이 있는 폴더에 새로 쓴다.
        sidecar_path = os.path.join(
            _server_backup_dir(server, backup_name), f"{backup_name}_outfit_prompt.json")
    positive = ""
    sidecar_tags = []
    if os.path.isfile(sidecar_path):
        try:
            with open(sidecar_path, "r", encoding="utf-8") as handle:
                saved = json.load(handle)
            positive = str(saved.get("positive") or "")
            tags = saved.get("outfit_tags")
            sidecar_tags = [str(tag) for tag in tags] if isinstance(tags, list) else []
        except (OSError, ValueError):
            positive = ""
    if not positive.strip() and prompt_path:
        positive, _negative = server._extract_prompts_from_backup(prompt_path)
    if not positive.strip():
        raise ValueError("감정 이미지의 원본 프롬프트를 찾을 수 없습니다.")
    return positive, sidecar_path, sidecar_tags


def _emotion_asset_old_tags(server, backup_name):
    info_path = _find_backup_file(server, backup_name, "_info.json")
    try:
        with open(info_path, "r", encoding="utf-8") as handle:
            info = json.load(handle)
        return ((info.get("emotion_asset") or {}).get("outfit_tags") or [])
    except (OSError, ValueError, AttributeError):
        return []


def _load_emotion_submitted_prompt(server, backup_name):
    """이미지를 만든 그대로의 제출 문서(설정·레퍼런스·복장)를 읽는다.

    재생성은 이 문서를 그대로 다시 제출하고 시드만 새로 뽑아야 원본과 같은
    LoRA/SNAPREF/FACE 설정과 참조 이미지로 다시 그린다.
    """
    path = _find_backup_file(server, backup_name, "_submitted_prompt.json")
    if not path:
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            document = json.load(handle)
    except (OSError, ValueError) as exc:
        print(f"[EMOTION_REGEN] 제출 프롬프트 읽기 실패: path={path!r}, error={exc}")
        return None
    if not (isinstance(document, dict) and document.get("schema") == "soya.illustration.v2"):
        return None
    # Outfit cover regeneration records the swapped *_CONTENT text in its
    # sidecar; apply it so the next regeneration starts from the latest outfit.
    sidecar_path = (_find_backup_file(server, backup_name, "_outfit_prompt.json")
                    or os.path.join(_server_backup_dir(server, backup_name),
                                    f"{backup_name}_outfit_prompt.json"))
    if os.path.isfile(sidecar_path):
        try:
            with open(sidecar_path, "r", encoding="utf-8") as handle:
                sidecar = json.load(handle)
        except (OSError, ValueError):
            sidecar = {}
        overrides = sidecar.get("content_overrides") if isinstance(sidecar, dict) else None
        fields = document.get("fields")
        if isinstance(overrides, dict) and isinstance(fields, dict):
            for key, value in overrides.items():
                if str(key).endswith("_CONTENT") and isinstance(value, str):
                    fields[key] = value
    return document


def _emotion_prompt_text(server, backup_name):
    """Return the exact positive/negative text recorded for an emotion image."""
    document = _load_emotion_submitted_prompt(server, backup_name)
    fields = document.get("fields") if isinstance(document, dict) else None
    fields = fields if isinstance(fields, dict) else {}
    positive = str(fields.get("ANIMA_CONTENT") or fields.get("SDXL_CONTENT") or "").strip()
    negative = str(fields.get("NEGATIVE_ANIMA") or fields.get("NEGATIVE_SDXL") or "").strip()
    if not positive:
        try:
            positive = _load_emotion_prompt(server, backup_name)[0].strip()
        except ValueError:
            positive = ""
    return {"positive": positive, "negative": negative}


def _replace_document_outfit_tags(document, old_tags, new_tags):
    """제출 문서의 프롬프트 필드 안에서만 복장 태그를 교체한다.

    설정·레퍼런스·LoRA는 건드리지 않고, 장면 텍스트가 담긴 *_CONTENT 필드만
    바꿔 같은 구도에서 복장만 달라지게 한다.
    """
    updated = deepcopy(document)
    fields = updated.get("fields") if isinstance(updated, dict) else None
    if not isinstance(fields, dict):
        return updated
    for key, value in list(fields.items()):
        if str(key).endswith("_CONTENT") and isinstance(value, str):
            fields[key] = _replace_outfit_tags(value, old_tags, new_tags)
    return updated


# 그리기 설정 값이 그대로 V2 필드가 되는 항목. 프롬프트 텍스트(LoRA·프리셋이
# 조립된 *_CONTENT)는 재생성의 구도 보존을 위해 건드리지 않는다.
_DRAWING_MODE_BOOLEAN_FIELDS = {
    "hrf_activate": "HRF_ACTIVATE",
    "anima_hrf_activate": "ANIMA_HRF_ACTIVATE",
    "hrf_restore_size": "HRF_RESTORE_SIZE",
    "anima_fd_activate": "ANIMA_FD_ACTIVATE",
    "anima_hd_activate": "ANIMA_HD_ACTIVATE",
    "anima_ed_activate": "ANIMA_ED_ACTIVATE",
    "fd_activate": "FD_ACTIVATE",
    "hd_activate": "HD_ACTIVATE",
    "ed_activate": "ED_ACTIVATE",
}
_DRAWING_MODE_SNAPREF_FIELDS = {
    "snapref_enabled": "enabled",
    "snapref_preprocess": "preprocess",
    "snapref_strength": "strength",
    "snapref_strength_scale": "strength_scale",
    "snapref_fidelity": "fidelity",
    "snapref_secondary_reference": "secondary_reference",
    "snapref_mode": "mode",
    "snapref_megapixels": "megapixels",
}


def _emotion_mode_settings(server, bot_name, generation_mode):
    """현재 감정 봇의 지정한 생성 모드 그리기 설정을 읽는다."""
    token = storage.set_bot_scope("emotion")
    try:
        data = storage._load_bot_data()
    finally:
        storage.reset_bot_scope(token)
    bot = next(
        (entry for entry in data.get("bots", []) if entry.get("name") == bot_name),
        None,
    )
    if bot is None:
        raise ValueError("봇을 찾을 수 없습니다.")
    # resolve_illust_settings는 BotMode 인스턴스 메서드가 아니라 모듈 함수다.
    return storage.resolve_illust_settings(
        bot, "solo", generation_mode=generation_mode)


def _character_auto_settings(server, bot_name):
    """현재 감정 봇의 캐릭터 자동 생성 및 복장 이미지 생성 모드 설정."""
    return _emotion_mode_settings(server, bot_name, "character_auto")


def _illustration_settings(server, bot_name):
    """현재 감정 봇의 에셋 생성 모드 그리기 설정을 읽는다."""
    return _emotion_mode_settings(server, bot_name, "illustration")


def _is_profile_base_outfit(server, bot_name, character_name, profile_id, outfit):
    """복장이 프로필 카드의 기본(첫 번째) 복장인지 확인한다.

    프로필별 복장 이미지 생성이 만드는 기본 복장은 프로필 자신을 나타내고,
    이후에 추가되는 폴더는 그 프로필의 변주 복장이다. 카드를 찾지 못하면
    기존 동작을 유지하도록 변주 복장으로 취급한다.
    """
    token = storage.set_bot_scope("emotion")
    try:
        data = storage._load_bot_data()
    finally:
        storage.reset_bot_scope(token)
    bot = next(
        (entry for entry in data.get("bots", []) if entry.get("name") == bot_name),
        None,
    )
    if bot is None:
        raise ValueError("봇을 찾을 수 없습니다.")
    character = next(
        (entry for entry in bot.get("characters", [])
         if isinstance(entry, dict)
         and str(entry.get("name") or "").casefold() == character_name.casefold()),
        None,
    )
    if character is None:
        return False
    cards, _source = effective_character_cards(character, None)
    card = next(
        (entry for entry in cards
         if str(entry.get("id") or "").strip() == profile_id),
        None,
    )
    if card is None:
        card = next(
            (entry for entry in cards
             if str(entry.get("label") or "").casefold() == profile_id.casefold()),
            None,
        )
    if card is None:
        return False
    folders = _emotion_store_outfit_names(
        server, bot_name, character_name, profile_id)
    _label, base_folder = _profile_base_outfit(card, folders)
    return base_folder.casefold() == outfit.casefold()


def _apply_drawing_mode(document, settings):
    """제출 문서에 현재 그리기 설정을 덮어쓴다.

    복장 재생성은 구도와 참조 이미지를 유지하려고 원본 제출 문서를 다시
    제출한다. 그대로 두면 생성 당시의 그리기 설정이 계속 이기므로, 복장에
    맞는 생성 모드(기본 복장은 에셋 생성 모드, 변주 복장은 캐릭터 자동 생성
    및 복장 이미지 생성 모드)의 값으로 맞춘다. SNAPREF의
    asset/face_asset 쌍은 그대로 두어 워크플로 노드가 얼굴만/단계별 전달을
    다시 고를 수 있게 한다.
    """
    updated = deepcopy(document)
    fields = updated.get("fields") if isinstance(updated, dict) else None
    if not isinstance(fields, dict):
        return updated
    if "img_w" in settings:
        fields["IMG_W"] = int(settings["img_w"])
    if "img_h" in settings:
        fields["IMG_H"] = int(settings["img_h"])
    if "hrf_size" in settings:
        fields["HRF_SIZE"] = float(settings["hrf_size"])
    for setting_key, field_key in _DRAWING_MODE_BOOLEAN_FIELDS.items():
        if setting_key in settings:
            fields[field_key] = bool(settings[setting_key])
    snapref = fields.get("SNAPREF")
    if not isinstance(snapref, dict):
        snapref = {}
        fields["SNAPREF"] = snapref
    for setting_key, field_key in _DRAWING_MODE_SNAPREF_FIELDS.items():
        if setting_key in settings:
            snapref[field_key] = deepcopy(settings[setting_key])
    return updated


def register_emotion_outfit_regeneration(server):
    """Regenerate one emotion outfit cover with only its outfit tags changed."""
    runs = {}
    tasks = {}
    _track_emotion_activity(server, runs)

    def _cover_path(bot_name, char_name, profile, outfit):
        token = storage.set_bot_scope("emotion")
        try:
            root = os.path.abspath(os.fspath(storage.BOT_DIR))
        finally:
            storage.reset_bot_scope(token)
        parts = [str(value or "").strip() for value in (bot_name, char_name, profile, outfit)]
        if not all(parts):
            raise ValueError("봇, 캐릭터, 프로필, 복장을 모두 지정하세요.")
        outfit_dir = os.path.abspath(os.path.join(root, *parts))
        try:
            if os.path.commonpath([root, outfit_dir]) != root:
                raise ValueError("잘못된 복장 경로입니다.")
        except ValueError:
            raise ValueError("잘못된 복장 경로입니다.")
        if not os.path.isdir(outfit_dir):
            raise ValueError("복장 폴더를 찾을 수 없습니다.")
        images = server.bot_mode._emotion_image_files(outfit_dir)
        # 대표 표정이 지정된 복장은 그 감정 폴더의 이미지를 우선 재생성한다.
        cover_rel = server.bot_mode._emotion_outfit_cover_rel(
            bot_name, char_name, profile, outfit, images)
        if not cover_rel:
            raise ValueError("복장 대표 이미지가 없습니다.")
        for rel, path, _stat in images:
            if rel == cover_rel:
                return path
        raise ValueError("복장 대표 이미지를 찾을 수 없습니다.")

    async def regenerate(
        run, cover_path, backup_name, positive, sidecar_path, old_tags, outfit_tags, document,
    ):
        try:
            # 삽화/감정 봇 이름이 같아도 워크플로 백업 폴더는 하나를 공유한다.
            # workshop을 넘기지 않으면 재생성 큐가 삽화 흐름으로 기록되고 새 백업도
            # '삽화 백업'으로 분류되므로, 감정 삽화에서 시작한 작업임을 명시한다.
            body = {
                "name": backup_name,
                "workshop": "emotion",
                "illustration_flow_auto_open": False,
            }
            updated_document = None
            if document is not None:
                # Keep the original submission's settings and reference images;
                # only the *_CONTENT prompt fields carry the new outfit tags.
                updated_document = _replace_document_outfit_tags(
                    document, old_tags, outfit_tags)
                body["illustration_prompt_json"] = updated_document
            else:
                body["positive_override"] = positive
            result = await server.handle_api_regenerate(
                None,
                _body=body,
                _return_queue_result=True,
            )
            if not isinstance(result, dict) or result.get("error"):
                detail = result.get("error") if isinstance(result, dict) else getattr(result, "status", "")
                raise RuntimeError(f"재생성 요청을 처리하지 못했습니다: {detail}")
            image_bytes = result.get("image_bytes")
            if not image_bytes:
                raise RuntimeError("재생성한 이미지가 없습니다.")
            await asyncio.to_thread(_write_emotion_image, cover_path, image_bytes)
            # Keep the next regeneration on the prompt that now matches the cover.
            sidecar_payload = {
                "positive": positive,
                "outfit_tags": list(outfit_tags or []),
            }
            if updated_document is not None:
                sidecar_payload["content_overrides"] = {
                    key: value
                    for key, value in updated_document.get("fields", {}).items()
                    if str(key).endswith("_CONTENT") and isinstance(value, str)
                }
            await asyncio.to_thread(
                storage.write_json_atomic, sidecar_path, sidecar_payload)
            run.update(status="completed", message="복장 대표 이미지를 재생성했습니다.")
        except asyncio.CancelledError:
            run.update(status="cancelled", message="", error="복장 이미지 재생성이 중단되었습니다.")
        except Exception as exc:
            traceback.print_exc()
            run.update(status="failed", message="", error=f"복장 이미지 재생성에 실패했습니다: {exc}")
        finally:
            tasks.pop(
                (run["bot_name"], run["char_name"], run["profile"], run["outfit"]),
                None,
            )

    async def handle(request):
        if request.method == "GET":
            key = (
                request.query.get("bot_name", ""),
                request.query.get("char_name", ""),
                request.query.get("profile", ""),
                request.query.get("outfit", ""),
            )
            return web.json_response({"run": runs.get(key)})
        try:
            body = await request.json()
            if not isinstance(body, dict) or any(
                not isinstance(body.get(key), str) or not body[key]
                for key in ("bot_name", "char_name", "profile", "outfit")
            ):
                raise ValueError("재생성할 복장을 선택하세요.")
            key = (
                body["bot_name"].strip(),
                body["char_name"].strip(),
                body["profile"].strip(),
                body["outfit"].strip(),
            )
            if key in tasks:
                return web.json_response({"run": runs[key]}, status=202)
            cover_path = _cover_path(*key)
            backup_name = _find_emotion_asset_backup(server, cover_path)
            if not backup_name:
                raise ValueError("이 복장 이미지의 생성 기록이 없어 재생성할 수 없습니다.")
            positive, sidecar_path, sidecar_tags = _load_emotion_prompt(server, backup_name)
            old_tags = body.get("old_tags") if isinstance(body.get("old_tags"), list) else []
            new_tags = body.get("new_tags") if isinstance(body.get("new_tags"), list) else []
            if not old_tags:
                # Prefer the tags of the prompt the sidecar now holds, then the backup record.
                old_tags = sidecar_tags or _emotion_asset_old_tags(server, backup_name)
            # 첫 번째 복장은 프로필 자신이므로 에셋 생성 모드, 두 번째
            # 복장부터는 변주 복장이므로 캐릭터 자동 생성 및 복장 이미지
            # 생성 모드의 설정을 따른다. 복장 태그만 바꿔 재생성해도
            # 크기·레퍼런스 전달·디테일러 설정이 생성 당시 값에 고정되지 않는다.
            document = _load_emotion_submitted_prompt(server, backup_name)
            if document is not None:
                base_outfit = _is_profile_base_outfit(
                    server, key[0], key[1], key[2], key[3])
                settings = (
                    _illustration_settings(server, key[0])
                    if base_outfit
                    else _character_auto_settings(server, key[0]))
                document = _apply_drawing_mode(document, settings)
            run = {
                "bot_name": key[0],
                "char_name": key[1],
                "profile": key[2],
                "outfit": key[3],
                "id": uuid.uuid4().hex,
                "status": "generating",
                "message": "복장 대표 이미지 재생성 중…",
                "error": "",
            }
            runs[key] = run
            tasks[key] = asyncio.create_task(
                regenerate(
                    run, cover_path, backup_name,
                    _replace_outfit_tags(positive, old_tags, new_tags), sidecar_path,
                    old_tags, new_tags, document,
                )
            )
            return web.json_response({"run": run}, status=202)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            return web.json_response({"error": str(exc)}, status=400)

    async def cleanup(_app):
        _EMOTION_ACTIVITY_STORES.pop(id(server), None)
        pending = list(tasks.values())
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

    server.app.router.add_get("/api/bot_mode/regenerate_emotion_outfit", handle)
    server.app.router.add_post("/api/bot_mode/regenerate_emotion_outfit", handle)
    server.app.on_cleanup.append(cleanup)


def _record_emotion_outfit_tags(server, backup_name, outfit_tags):
    """Record the outfit tags embedded in a regenerated emotion backup.

    The new backup inherits its emotion_asset metadata from the source backup,
    so without this the next regeneration would look for the previous outfit
    tags in a prompt that already carries the new ones and change nothing.
    """
    if not backup_name or not outfit_tags:
        return
    try:
        path = _find_backup_file(server, backup_name, "_info.json")
        if not path:
            return
        with open(path, "r", encoding="utf-8") as handle:
            info = json.load(handle)
        asset = info.get("emotion_asset") if isinstance(info, dict) else None
        if not isinstance(asset, dict):
            return
        asset["outfit_tags"] = [str(tag) for tag in outfit_tags]
        storage.write_json_atomic(path, info)
    except Exception as exc:
        # Bookkeeping only: the image was already replaced, so never fail the
        # run because the tag cache could not be updated.
        print(
            f"[EMOTION_REGEN] 복장 태그 기록 실패: "
            f"backup={backup_name!r}, error={exc}"
        )


def register_emotion_image_regeneration(server):
    """Regenerate one emotion image, keeping its scene and refreshing the outfit tags."""
    runs = {}
    tasks = {}
    _track_emotion_activity(server, runs)

    def _image_path(bot_name, char_name, profile, outfit, emotion):
        token = storage.set_bot_scope("emotion")
        try:
            folder, error = server.bot_mode._emotion_image_dir(
                bot_name, char_name, profile, outfit, emotion)
            images = server.bot_mode._emotion_flat_images(folder) if folder else []
        finally:
            storage.reset_bot_scope(token)
        if folder is None:
            raise ValueError(error)
        if not images:
            raise ValueError("감정 폴더에 이미지가 없습니다.")
        # The list uses the first image as the emotion cover, so refresh that one.
        return os.path.join(folder, images[0])

    async def regenerate(run, image_path, backup_name, positive, document, old_tags, new_tags):
        try:
            # 복장 대표 이미지 재생성도 같은 이유로 워크숍을 고정한다.
            # (백업 저장소 공유 · 삽화/감정 봇 이름 중복)
            body = {
                "name": backup_name,
                "workshop": "emotion",
                "illustration_flow_auto_open": False,
            }
            if document is not None:
                # Keep the stored scene, drawing settings and embedded reference
                # images, but re-read the outfit: only the *_CONTENT prompt
                # fields get the card's current outfit tags. The seed is still
                # re-rolled by handle_api_regenerate.
                body["illustration_prompt_json"] = _replace_document_outfit_tags(
                    document, old_tags, new_tags)
            elif positive:
                body["positive_override"] = _replace_outfit_tags(
                    positive, old_tags, new_tags)
            result = await server.handle_api_regenerate(
                None,
                _body=body,
                _return_queue_result=True,
            )
            if not isinstance(result, dict) or result.get("error"):
                detail = result.get("error") if isinstance(result, dict) else getattr(result, "status", "")
                raise RuntimeError(f"재생성 요청을 처리하지 못했습니다: {detail}")
            image_bytes = result.get("image_bytes")
            if not image_bytes:
                raise RuntimeError("재생성한 이미지가 없습니다.")
            await asyncio.to_thread(_write_emotion_image, image_path, image_bytes)
            # Keep the next regeneration's old-tag lookup in step with the new
            # outfit tags now stored in this backup's prompt.
            await asyncio.to_thread(
                _record_emotion_outfit_tags,
                server,
                str(result.get("backup_name") or ""),
                new_tags,
            )
            run.update(status="completed", message="감정 이미지를 재생성했습니다.")
        except asyncio.CancelledError:
            run.update(status="cancelled", message="", error="감정 이미지 재생성이 중단되었습니다.")
        except Exception as exc:
            traceback.print_exc()
            run.update(status="failed", message="", error=f"감정 이미지 재생성에 실패했습니다: {exc}")
        finally:
            tasks.pop(
                (run["bot_name"], run["char_name"], run["profile"], run["outfit"], run["emotion"]),
                None,
            )

    async def handle(request):
        if request.method == "GET":
            key = tuple(
                request.query.get(name, "")
                for name in ("bot_name", "char_name", "profile", "outfit", "emotion")
            )
            return web.json_response({"run": runs.get(key)})
        try:
            body = await request.json()
            required = ("bot_name", "char_name", "profile", "outfit", "emotion")
            if not isinstance(body, dict) or any(
                not isinstance(body.get(name), str) or not body[name] for name in required
            ):
                raise ValueError("재생성할 감정을 선택하세요.")
            key = tuple(body[name].strip() for name in required)
            if key in tasks:
                return web.json_response({"run": runs[key]}, status=202)
            image_path = _image_path(*key)
            backup_name = _find_emotion_asset_backup(server, image_path)
            if not backup_name:
                raise ValueError("이 감정 이미지의 생성 기록이 없어 재생성할 수 없습니다.")
            document = _load_emotion_submitted_prompt(server, backup_name)
            try:
                positive, _sidecar_path, sidecar_tags = _load_emotion_prompt(
                    server, backup_name)
            except ValueError:
                positive, sidecar_tags = "", []
            old_tags = body.get("old_tags") if isinstance(body.get("old_tags"), list) else []
            if not old_tags:
                old_tags = sidecar_tags or _emotion_asset_old_tags(server, backup_name)
            requested_tags = body.get("new_tags") if isinstance(body.get("new_tags"), list) else []
            new_tags = [str(tag).strip() for tag in requested_tags if str(tag).strip()]
            if not new_tags:
                # Re-read the outfit saved on the character card, so regenerating
                # follows outfit edits even when the caller sends no tag list.
                new_tags = _emotion_card_outfit_tags(server, *key[:4])
            run = {
                "bot_name": key[0],
                "char_name": key[1],
                "profile": key[2],
                "outfit": key[3],
                "emotion": key[4],
                "id": uuid.uuid4().hex,
                "status": "generating",
                "message": "감정 이미지 재생성 중…",
                "error": "",
            }
            runs[key] = run
            tasks[key] = asyncio.create_task(
                regenerate(
                    run, image_path, backup_name,
                    positive, document, old_tags, new_tags)
            )
            return web.json_response({"run": run}, status=202)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            return web.json_response({"error": str(exc)}, status=400)

    async def read_prompt(request):
        """GET prompt used for the emotion cover image shown in the lightbox."""
        try:
            key = tuple(
                request.query.get(name, "").strip()
                for name in ("bot_name", "char_name", "profile", "outfit", "emotion")
            )
            if any(not value for value in key):
                raise ValueError("감정을 선택하세요.")
            image_path = _image_path(*key)
            backup_name = _find_emotion_asset_backup(server, image_path)
            if not backup_name:
                raise ValueError("이 감정 이미지의 생성 기록이 없습니다.")
            return web.json_response({"prompt": _emotion_prompt_text(server, backup_name)})
        except (ValueError, OSError, KeyError, TypeError) as exc:
            return web.json_response({"error": str(exc)}, status=400)

    async def cleanup(_app):
        _EMOTION_ACTIVITY_STORES.pop(id(server), None)
        pending = list(tasks.values())
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

    async def activity(request):
        """GET count of in-flight emotion jobs for the 캐릭터 panel header."""
        return web.json_response(
            _emotion_activity(server, request.query.get("bot_name", "").strip()))

    server.app.router.add_get("/api/bot_mode/regenerate_emotion_image", handle)
    server.app.router.add_post("/api/bot_mode/regenerate_emotion_image", handle)
    server.app.router.add_get("/api/bot_mode/emotion_image_prompt", read_prompt)
    server.app.router.add_get("/api/bot_mode/emotion_regeneration_activity", activity)
    server.app.on_cleanup.append(cleanup)


def _emotion_bulk_outfit_tags(card, outfit):
    """Return the drawable tags for one outfit folder.

    The card's saved outfit pool is the source of truth. Legacy folders whose
    label was auto-built from a tag list fall back to that comma-separated name,
    matching the emotion pipeline's unsaved-outfit fallback.
    """
    if not isinstance(card, dict):
        return []
    wanted = pipeline._emotion_outfit_tag_key(outfit)
    for entry in card.get("outfit_presets") or []:
        if not isinstance(entry, dict):
            continue
        if pipeline._emotion_outfit_tag_key(entry.get("label")) != wanted:
            continue
        try:
            tags = tag_values(entry.get("tags") or [])
        except Exception:
            tags = []
        if tags:
            return tags
    return [part.strip() for part in str(outfit or "").split(",") if part.strip()][:12]


def _emotion_card_outfit_tags(server, bot_name, char_name, profile, outfit):
    """Read the outfit tags currently saved on the emotion card.

    Regeneration must re-read the outfit instead of replaying the tags baked
    into the stored prompt, otherwise outfit edits on the card never reach the
    regenerated emotion image.
    """
    token = storage.set_bot_scope("emotion")
    try:
        data = storage._load_bot_data()
        bot = next(
            (entry for entry in data.get("bots", []) if entry.get("name") == bot_name),
            None,
        )
        if bot is None:
            return []
        character = next(
            (entry for entry in bot.get("characters", [])
             if str(entry.get("name") or "").casefold() == char_name.casefold()),
            None,
        )
        if character is None:
            return []
        cards, _source = effective_character_cards(character, None)
        card = next(
            (entry for entry in cards if str(entry.get("id") or "") == profile),
            None,
        )
        return _emotion_bulk_outfit_tags(card, outfit) if card else []
    finally:
        storage.reset_bot_scope(token)


def _pending_emotion_presets(options, reuse_by_folder, *, include_existing):
    """Resolve preset rows to emotion folders, skipping duplicates and existing images."""
    pending = []
    seen = set()
    for option in options or []:
        if not isinstance(option, dict):
            continue
        name = str(option.get("name") or "").strip()
        folder = pipeline._emotion_folder_name(name)
        key = folder.casefold()
        if not folder or key in seen:
            continue
        seen.add(key)
        reuse_path = str(reuse_by_folder.get(key) or "")
        if reuse_path and not include_existing:
            continue
        pending.append({
            "id": str(option.get("id") or ""),
            "rating": str(option.get("rating") or "sfw"),
            "name": name,
            "folder": folder,
            "tags": list(option.get("tags") or []),
            "reuse_path": reuse_path,
        })
    return pending


def register_emotion_bulk_generation(server):
    """Generate every configured emotion preset for one outfit in one batch.

    The prompt is rebuilt from the emotion workshop's prompt files and the
    SFW/NSFW emotion preset rows, so the buttons follow 삽화 프롬프트 설정.
    Existing emotions are overwritten only in ``mode="all"``.
    """
    runs = {}
    tasks = {}
    _track_emotion_activity(server, runs)

    def _resolve(bot_name, char_name, profile, outfit):
        token = storage.set_bot_scope("emotion")
        try:
            data = storage._load_bot_data()
            bot = next(
                (entry for entry in data.get("bots", []) if entry.get("name") == bot_name),
                None,
            )
            if bot is None:
                raise ValueError("봇을 찾을 수 없습니다.")
            character = next(
                (entry for entry in bot.get("characters", [])
                 if str(entry.get("name") or "").casefold() == char_name.casefold()),
                None,
            )
            if character is None:
                raise ValueError("캐릭터를 찾을 수 없습니다.")
            cards, _source = effective_character_cards(character, None)
            card = next(
                (entry for entry in cards if str(entry.get("id") or "") == profile),
                None,
            )
            if card is None:
                raise ValueError("감정을 생성할 프로필 카드를 찾을 수 없습니다.")
            outfit_dir, error = server.bot_mode._emotion_image_dir(
                bot_name, char_name, profile, outfit, "")
            if error:
                raise ValueError(error)
            if not outfit_dir or not os.path.isdir(outfit_dir):
                raise ValueError("복장 폴더를 찾을 수 없습니다.")
            outfit_tags = _emotion_bulk_outfit_tags(card, outfit)
            if not outfit_tags:
                raise ValueError(
                    "이 복장의 태그를 찾을 수 없습니다. 복장 태그를 저장한 뒤 다시 실행하세요.")
            return card, outfit_tags
        finally:
            storage.reset_bot_scope(token)

    def _prompt_job(item, asset, slot, prompts, toggles, snapshot):
        descriptor = scene_pipeline._emotion_asset_descriptor(
            asset, slot, prompts, toggles)
        positive = str(descriptor.get("raw_positive") or "")
        negative = str(descriptor.get("raw_negative") or "")
        if not positive.strip():
            raise RuntimeError("감정 이미지 프롬프트가 비어 있습니다.")
        provider = str(snapshot.get("provider") or "comfy").strip().lower()
        workflow_type = str(snapshot.get("illustration_workflow_type") or "")
        prompt_format = workflow_profiles.illustration_prompt_format(
            workflow_type, "comfy" if provider == "hybrid" else provider)
        prompt_id = f"emotion-bulk-{uuid.uuid4().hex}"
        prompt_data = {
            "positive": {"class_type": "STRING", "_meta": {"title": "긍정프롬프트"}, "inputs": {"value": positive}},
            "negative": {"class_type": "STRING", "_meta": {"title": "부정프롬프트"}, "inputs": {"value": negative}},
        }
        params = {
            "prompt_id": prompt_id,
            "prompt_data": prompt_data,
            "provider": provider,
            "workshop": "emotion",
            "illustration_flow_skip": True,
            "raw_body": {
                "illustration_provider": provider,
                "illustration_prompt_format": prompt_format,
                "illustration_defer_postprocess": True,
                "illustration_gen_method": "감정 에셋",
                "illustration_visual_states": server._descriptor_visual_states(descriptor),
            },
        }
        if provider == "hybrid":
            params["hybrid_prompt_formats"] = {
                value: workflow_profiles.illustration_prompt_format(workflow_type, value)
                for value in ("comfy", "chansub")
            }
        server.prompts[prompt_id] = {
            "status": "running", "prompt": prompt_data, "client_id": "", "extra_data": {},
            "outputs": {}, "filename": None, "save_node_id": None, "image_bytes": None,
            "timestamp": time.time(), "_illustration_runtime_snapshot": snapshot,
        }
        return {
            "item": item, "prompt_id": prompt_id, "params": params,
            "positive": positive, "negative": negative, "queue_item": None,
        }

    async def _finish_job(run, job, snapshot):
        item = job["item"]
        try:
            try:
                await job["queue_item"].completion_future
            except RuntimeError:
                # The work queue lets the user cancel individual batch items.
                if str(getattr(job["queue_item"], "status", "")) == "cancelled":
                    run["cancelled"].append(item["name"])
                    return
                raise
            entry = server.prompts.get(job["prompt_id"], {})
            pixels = entry.get("_deferred_image_bytes")
            if not pixels:
                raise RuntimeError(entry.get("error") or "생성한 감정 이미지가 없습니다.")
            saved_path = item["reuse_path"]
            if saved_path:
                await asyncio.to_thread(_write_emotion_image, saved_path, pixels)
                run["updated"].append(item["name"])
            else:
                target_dir, _reuse = pipeline._emotion_asset_store_paths(
                    run["bot_name"], run["char_name"], run["profile"],
                    run["outfit"], item["folder"])
                saved_path = await asyncio.to_thread(
                    storage.save_emotion_asset_bytes, target_dir, pixels)
                if not saved_path:
                    raise RuntimeError("감정 이미지를 저장하지 못했습니다.")
                run["created"].append(item["name"])
            state = entry.get("_deferred_finalize")
            await _save_backup(
                run, item, pixels, job["prompt_id"],
                job["positive"], job["negative"],
                state if isinstance(state, dict) else {}, saved_path, snapshot)
        finally:
            server._discard_deferred_illustration_prompt(job["prompt_id"], "감정 일괄 생성 종료")

    async def _save_backup(run, item, pixels, prompt_id, positive, negative, state, saved_path, snapshot):
        save_backup = getattr(server.queue_manager, "save_backup", None)
        if not callable(save_backup):
            return
        try:
            await _write_backup(
                run, item, pixels, prompt_id, positive, negative, state,
                saved_path, snapshot, save_backup)
        except Exception as exc:
            traceback.print_exc()
            run["warnings"].append(f"{item['name']}: 백업을 저장하지 못했습니다: {exc}")

    async def _write_backup(
            run, item, pixels, prompt_id, positive, negative, state,
            saved_path, snapshot, save_backup):
        # Record the exact prompt the generator used, including presets/references.
        positive = str(state.get("positive") or positive)
        negative = str(state.get("negative") or negative)
        conversion_info = deepcopy(state.get("conversion_info") or {})
        conversion_info["emotion_asset"] = {
            "anchor": "",
            "character": run["char_name"],
            "profile_id": run["profile"],
            "outfit": run["outfit"],
            "outfit_folder": run["outfit"],
            "outfit_tags": list(run["outfit_tags"]),
            "emotion_id": item["id"],
            "emotion_name": item["name"],
            "rating": item["rating"],
            "target_dir": os.path.dirname(saved_path),
            "saved_path": saved_path,
        }
        provider = str(state.get("provider") or snapshot.get("provider") or "comfy")
        await save_backup(
            pixels, prompt_id, positive, negative,
            generation_time=state.get("generation_time"),
            bot_name=run["bot_name"],
            gen_method=str(state.get("gen_method") or "감정 에셋"),
            workshop="emotion",
            provider=provider,
            provider_mode=str(state.get("provider_mode") or provider),
            prompt_provider=str(state.get("prompt_provider") or provider),
            generation_params=state.get("generation_params"),
            original_workflow_snapshot=state.get("original_workflow"),
            api_workflow_snapshot=state.get("api_workflow"),
            conversion_info_snapshot=conversion_info,
            submitted_prompt_snapshot=state.get("submitted_prompt"),
            illustration_visual_states=state.get("illustration_visual_states"),
            execution_target=str(state.get("execution_target") or ""),
        )

    async def _cancel_jobs(jobs):
        """Cancel batch items still waiting in the work queue."""
        for job in jobs:
            queue_item = job.get("queue_item")
            if queue_item is None or queue_item.completion_future.done():
                continue
            try:
                await server.queue_manager.cancel_item(queue_item.id)
            except Exception:
                traceback.print_exc()

    async def collect(run, jobs, snapshot):
        token = storage.set_bot_scope("emotion")
        try:
            # The existing work queue already holds the whole batch. This task
            # only saves each finished image and records cancellations.
            for job in jobs:
                run["current"] = job["item"]["name"]
                try:
                    await _finish_job(run, job, snapshot)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    traceback.print_exc()
                    run["errors"].append(f"{job['item']['name']}: {exc}")
                finally:
                    run["done"] += 1
            if run["created"] or run["updated"]:
                parts = []
                if run["created"]:
                    parts.append(f"새 감정 {len(run['created'])}개")
                if run["updated"]:
                    parts.append(f"기존 감정 {len(run['updated'])}개 교체")
                if run["cancelled"]:
                    parts.append(f"취소 {len(run['cancelled'])}개")
                run["status"] = "completed"
                run["message"] = " · ".join(parts) + " 완료"
                if run["errors"]:
                    run["error"] = f"{len(run['errors'])}개 실패: {run['errors'][0]}"
            elif run["errors"]:
                run["status"] = "failed"
                run["error"] = f"감정 이미지 생성에 실패했습니다: {run['errors'][0]}"
            elif run["cancelled"]:
                run["status"] = "cancelled"
                run["message"] = f"감정 {len(run['cancelled'])}개 생성을 취소했습니다."
            else:
                run["status"] = "completed"
                run["message"] = "생성할 감정이 없습니다."
        except asyncio.CancelledError:
            await _cancel_jobs(jobs)
            run["status"] = "cancelled"
            run["message"] = ""
            run["error"] = "감정 일괄 생성이 중단되었습니다."
        except Exception as exc:
            await _cancel_jobs(jobs)
            traceback.print_exc()
            run["status"] = "failed"
            run["message"] = ""
            run["error"] = f"감정 일괄 생성에 실패했습니다: {exc}"
        finally:
            for job in jobs:
                server._discard_deferred_illustration_prompt(
                    job["prompt_id"], "감정 일괄 생성 종료")
            run["current"] = ""
            tasks.pop(
                (run["bot_name"], run["char_name"], run["profile"], run["outfit"]),
                None,
            )
            storage.reset_bot_scope(token)

    async def handle(request):
        if request.method == "GET":
            key = tuple(
                request.query.get(name, "").strip()
                for name in ("bot_name", "char_name", "profile", "outfit")
            )
            return web.json_response({"run": runs.get(key)})
        try:
            if storage.bot_scope() != "emotion":
                raise ValueError("감정 삽화 공방에서만 감정 일괄 생성을 실행할 수 있습니다.")
            body = await request.json()
            required = ("bot_name", "char_name", "profile", "outfit")
            if not isinstance(body, dict) or any(
                not isinstance(body.get(name), str) or not body[name].strip()
                for name in required
            ):
                raise ValueError("감정을 생성할 복장을 선택하세요.")
            mode = str(body.get("mode") or "missing").strip().lower()
            if mode not in ("all", "missing"):
                raise ValueError("감정 생성 모드를 확인하세요.")
            key = tuple(body[name].strip() for name in required)
            action = str(body.get("action") or "").strip().lower()
            if action:
                if action != "cancel":
                    raise ValueError("지원하지 않는 감정 생성 작업입니다.")
                run = runs.get(key)
                if run is None:
                    raise ValueError("취소할 감정 일괄 생성 작업이 없습니다.")
                run["cancel_requested"] = True
                batch_id = str(run.get("batch_id") or "")
                cancelled = 0
                cancel_batch = getattr(server.queue_manager, "cancel_batch", None)
                if batch_id and callable(cancel_batch):
                    cancelled = await cancel_batch(batch_id)
                return web.json_response(
                    {"run": run, "cancelled": int(cancelled or 0)}, status=202)
            if key in tasks:
                return web.json_response({"run": runs[key]}, status=202)
            card, outfit_tags = _resolve(*key)
            snapshot = server._capture_illustration_runtime_snapshot(
                {**server.app_config, "emotion_bot_selected": key[0]},
                workshop="emotion",
            )
            toggles = snapshot["illustration_context_toggles"]
            options, warnings = pipeline._emotion_preset_options(toggles)
            reuse_by_folder = {}
            for option in options:
                folder = pipeline._emotion_folder_name(option.get("name"))
                if not folder:
                    continue
                _target, reuse_path = pipeline._emotion_asset_store_paths(
                    key[0], key[1], key[2], key[3], folder)
                reuse_by_folder[folder.casefold()] = reuse_path
            pending = _pending_emotion_presets(
                options, reuse_by_folder, include_existing=(mode == "all"))
            try:
                representative = server.bot_mode._emotion_outfit_representative(*key)
            except Exception:
                representative = None
            if representative:
                folder = pipeline._emotion_folder_name(representative.get("name")).casefold()
                pending.sort(
                    key=lambda row: 0 if row["folder"].casefold() == folder else 1)
            pending_folders = {row["folder"].casefold() for row in pending}
            skipped = [
                str(option.get("name") or "").strip()
                for option in options
                if str(option.get("name") or "").strip()
                and pipeline._emotion_folder_name(option.get("name")).casefold()
                not in pending_folders
            ]
            # Build every prompt and load the whole batch into the existing work
            # queue before returning, so the queue is populated immediately.
            prompts = pipeline.load_prompt_files("emotion")
            jobs, specs, failures = [], [], []
            for index, item in enumerate(pending, start=1):
                asset = {
                    "character": key[1],
                    "profile_id": key[2],
                    "gender": str(card.get("gender_tag") or ""),
                    "appearance": tag_values(card.get("appearance") or []),
                    "outfit_tags": list(outfit_tags),
                    "emotion_tags": list(item["tags"]),
                    "emotion_id": item["id"],
                    "emotion_name": item["name"],
                    "rating": item["rating"],
                    "emotion_folder": item["folder"],
                }
                if not item["tags"]:
                    warnings.append(
                        f"{item['name']}: 표정 프리셋 태그가 없어 표정 없이 생성합니다.")
                try:
                    job = _prompt_job(
                        item, asset, index, prompts, toggles, snapshot)
                    jobs.append(job)
                    specs.append({
                        "type": "illustration",
                        "label": f"감정 일괄 생성 · {key[1]} · {item['name']}",
                        "batch_label": f"감정 일괄 생성 · {key[1]} · {key[3]}",
                        "params": job["params"],
                    })
                except Exception as exc:
                    traceback.print_exc()
                    failures.append(f"{item['name']}: {exc}")
            try:
                created = await server.queue_manager.add_items_batch(
                    specs, priority=0) if specs else []
            except Exception as exc:
                for job in jobs:
                    server._discard_deferred_illustration_prompt(
                        job["prompt_id"], "감정 일괄 생성 등록 실패")
                raise ValueError(
                    f"작업 큐에 감정 생성을 등록하지 못했습니다: {exc}") from exc
            for job, queue_item in zip(jobs, created):
                job["queue_item"] = queue_item
            run = {
                "bot_name": key[0], "char_name": key[1], "profile": key[2],
                "outfit": key[3], "mode": mode,
                "outfit_tags": list(outfit_tags),
                "id": uuid.uuid4().hex, "status": "generating",
                "message": f"감정 이미지 {len(pending)}개 생성 중…" if pending else "",
                "error": "", "total": len(pending), "done": len(failures), "current": "",
                "created": [], "updated": [], "skipped": skipped, "warnings": list(warnings),
                "errors": list(failures), "cancelled": [], "cancel_requested": False,
                "batch_id": str(getattr(created[0], "batch_id", "") or "") if created else "",
            }
            runs[key] = run
            tasks[key] = asyncio.create_task(collect(run, jobs, snapshot))
            return web.json_response({"run": run}, status=202)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            return web.json_response({"error": str(exc)}, status=400)

    async def cleanup(_app):
        pending_tasks = list(tasks.values())
        for task in pending_tasks:
            task.cancel()
        await asyncio.gather(*pending_tasks, return_exceptions=True)

    server.app.router.add_get("/api/bot_mode/generate_emotion_images", handle)
    server.app.router.add_post("/api/bot_mode/generate_emotion_images", handle)
    server.app.on_cleanup.append(cleanup)


def _profile_base_outfit(card: dict, folders: list[str]) -> tuple[str, str]:
    """Pick the card's default outfit (see pipeline.emotion_profile_base_outfit)."""
    return pipeline.emotion_profile_base_outfit(card, folders)


_PROFILE_OUTFIT_PLAN_KEYS = (
    "character", "profile_id", "profile_label", "outfit",
    "emotion_name", "rating", "action", "reason", "warning",
)


def _public_outfit_plan(plan: dict) -> dict:
    """Keep filesystem paths and raw tag lists inside the worker."""
    return {
        key: plan[key] for key in _PROFILE_OUTFIT_PLAN_KEYS
        if plan.get(key) is not None
    }


def _emotion_store_outfit_names(server, bot_name, character_name, profile_id):
    """List the profile's outfit folders from the emotion store."""
    token = storage.set_bot_scope("emotion")
    try:
        return [
            str(entry.get("name") or "").strip()
            for entry in server.bot_mode._emotion_outfits(
                bot_name, character_name, profile_id)
            if str(entry.get("name") or "").strip()
        ]
    finally:
        storage.reset_bot_scope(token)


def _profile_outfit_plans(server, bot_name: str, toggles: dict) -> list[dict]:
    """Return one base-outfit image plan per profile card.

    The card's own default outfit and appearance tags define the image. The
    emotion bot's drawing settings are resolved later from the bot name in the
    runtime snapshot, so this never borrows another bot's or the illustration
    workshop's prompt presets.
    """
    token = storage.set_bot_scope("emotion")
    try:
        data = storage._load_bot_data()
    finally:
        storage.reset_bot_scope(token)
    bot = next(
        (entry for entry in data.get("bots", []) if entry.get("name") == bot_name),
        None,
    )
    if bot is None:
        raise ValueError("봇을 찾을 수 없습니다.")
    catalog = pipeline._emotion_expression_catalog()
    nsfw_enabled = bool(toggles.get("nsfw")) and bool(
        toggles.get("nsfw_asset_generation_enabled"))
    plans = []
    seen = set()
    for character in bot.get("characters", []):
        if not isinstance(character, dict):
            continue
        character_name = str(character.get("name") or "").strip()
        if not character_name:
            continue
        cards, _source = effective_character_cards(character, None)
        for card in cards:
            profile_id = str(card.get("id") or "").strip()
            if not profile_id:
                continue
            plan = {
                "character": character_name,
                "profile_id": profile_id,
                "profile_label": str(card.get("label") or profile_id),
                "outfit_tags": tag_values(card.get("default_outfit") or []),
                "gender": str(card.get("gender_tag") or ""),
                "appearance": tag_values(card.get("appearance") or []),
                "rating": "sfw",
                "emotion_name": "",
                "emotion_folder": "",
                "emotion_tags": [],
                "target_dir": "",
                "reuse_path": "",
                "outfit": "",
                "outfit_folder": "",
                "action": "skip",
                "reason": "",
            }
            if not plan["outfit_tags"]:
                plan["reason"] = "기본 복장 태그 없음"
                plans.append(plan)
                continue
            existing_folders = _emotion_store_outfit_names(
                server, bot_name, character_name, profile_id)
            label, outfit_folder = _profile_base_outfit(card, existing_folders)
            plan.update(outfit=label or "default outfit", outfit_folder=outfit_folder)
            if not outfit_folder:
                plan["reason"] = "복장 폴더 이름을 만들 수 없음"
                plans.append(plan)
                continue
            key = (character_name.casefold(), profile_id.casefold())
            if key in seen:
                plan["reason"] = "같은 프로필이 중복됨"
                plans.append(plan)
                continue
            seen.add(key)
            if pipeline._emotion_outfit_has_images(
                    bot_name, character_name, profile_id, outfit_folder):
                plan["reason"] = "이미 복장 이미지 있음"
                plans.append(plan)
                continue
            representative = server.bot_mode._emotion_outfit_representative(
                bot_name, character_name, profile_id, outfit_folder)
            if representative:
                rating = str(representative.get("rating") or "sfw").strip().lower()
                name = str(representative.get("name") or "").strip()
                if rating == "nsfw" and not nsfw_enabled:
                    plan["reason"] = "NSFW 복장 대표 표정 · NSFW 에셋 생성이 꺼져 있음"
                    plans.append(plan)
                    continue
                if not name:
                    plan["reason"] = "복장 대표 표정 이름 없음"
                    plans.append(plan)
                    continue
                preset = str(representative.get("preset") or "").strip()
                tags = catalog.get(preset) if preset else []
                if preset and tags is None:
                    plan["warning"] = (
                        f"복장 대표 표정 프리셋을 찾을 수 없음: {preset}")
                    tags = []
                plan["emotion_tags"] = [
                    str(tag).strip() for tag in (tags or []) if str(tag).strip()]
            else:
                rating, name = "sfw", "neutral"
            emotion_folder = pipeline._emotion_folder_name(name) or "neutral"
            target_dir, reuse_path = pipeline._emotion_asset_store_paths(
                bot_name, character_name, profile_id, outfit_folder, emotion_folder)
            if not target_dir:
                plan["reason"] = "저장 경로를 확인할 수 없음"
                plans.append(plan)
                continue
            if reuse_path:
                plan["reason"] = "이미 대표 이미지 있음"
                plans.append(plan)
                continue
            plan.update(
                action="generate", rating=rating,
                emotion_name=name, emotion_folder=emotion_folder,
                target_dir=target_dir, reuse_path=reuse_path)
            plans.append(plan)
    return plans


def _profile_outfit_prompt_job(server, plan, slot, prompts, toggles, snapshot):
    """Build one deferred emotion-asset job for a profile's base outfit image."""
    asset = {
        "character": plan["character"],
        "profile_id": plan["profile_id"],
        "gender": plan.get("gender") or "",
        "appearance": list(plan.get("appearance") or []),
        "outfit_tags": list(plan.get("outfit_tags") or []),
        "emotion_tags": list(plan.get("emotion_tags") or []),
        "emotion_id": "",
        "emotion_name": plan["emotion_name"],
        "rating": plan["rating"],
        "emotion_folder": plan["emotion_folder"],
    }
    descriptor = scene_pipeline._emotion_asset_descriptor(
        asset, slot, prompts, toggles)
    positive = str(descriptor.get("raw_positive") or "")
    negative = str(descriptor.get("raw_negative") or "")
    if not positive.strip():
        raise RuntimeError("복장 이미지 프롬프트가 비어 있습니다.")
    provider = str(snapshot.get("provider") or "comfy").strip().lower()
    workflow_type = str(snapshot.get("illustration_workflow_type") or "")
    prompt_format = workflow_profiles.illustration_prompt_format(
        workflow_type, "comfy" if provider == "hybrid" else provider)
    prompt_id = f"emotion-profile-outfit-{uuid.uuid4().hex}"
    prompt_data = {
        "positive": {"class_type": "STRING", "_meta": {"title": "긍정프롬프트"}, "inputs": {"value": positive}},
        "negative": {"class_type": "STRING", "_meta": {"title": "부정프롬프트"}, "inputs": {"value": negative}},
    }
    params = {
        "prompt_id": prompt_id,
        "prompt_data": prompt_data,
        "provider": provider,
        # 큐 실행 시 감정 삽화 봇 저장소/그리기 설정을 쓰도록 워크숍을 고정한다.
        # (두 공방의 봇 이름이 같아도 큐가 삽화 쪽으로 해석하지 않게)
        "workshop": "emotion",
        "illustration_flow_skip": True,
        "raw_body": {
            "illustration_provider": provider,
            "illustration_prompt_format": prompt_format,
            "illustration_defer_postprocess": True,
            "illustration_gen_method": "감정 에셋",
            "illustration_visual_states": server._descriptor_visual_states(descriptor),
        },
    }
    if provider == "hybrid":
        params["hybrid_prompt_formats"] = {
            value: workflow_profiles.illustration_prompt_format(workflow_type, value)
            for value in ("comfy", "chansub")
        }
    server.prompts[prompt_id] = {
        "status": "running", "prompt": prompt_data, "client_id": "", "extra_data": {},
        "outputs": {}, "filename": None, "save_node_id": None, "image_bytes": None,
        "timestamp": time.time(), "_illustration_runtime_snapshot": snapshot,
    }
    return {
        "plan": plan, "prompt_id": prompt_id, "params": params,
        "positive": positive, "negative": negative, "queue_item": None,
    }


def register_emotion_profile_outfit_generation(server):
    """Generate one default-outfit image for every profile card of a bot."""
    runs = {}
    tasks = {}
    _track_emotion_activity(server, runs)

    async def _finish_job(run, job, snapshot):
        plan = job["plan"]
        try:
            try:
                await job["queue_item"].completion_future
            except RuntimeError:
                if str(getattr(job["queue_item"], "status", "")) == "cancelled":
                    run["cancelled"].append(plan["profile_label"])
                    return
                raise
            entry = server.prompts.get(job["prompt_id"], {})
            pixels = entry.get("_deferred_image_bytes")
            if not pixels:
                raise RuntimeError(entry.get("error") or "생성한 복장 이미지가 없습니다.")
            saved_path = plan.get("reuse_path") or await asyncio.to_thread(
                storage.save_emotion_asset_bytes, plan["target_dir"], pixels)
            if not saved_path:
                raise RuntimeError("복장 이미지를 저장하지 못했습니다.")
            run["created"].append(f"{plan['character']} · {plan['profile_label']}")
            state = entry.get("_deferred_finalize")
            await _save_backup(
                run, job, pixels, job["prompt_id"],
                job["positive"], job["negative"],
                state if isinstance(state, dict) else {}, saved_path, snapshot)
        finally:
            server._discard_deferred_illustration_prompt(
                job["prompt_id"], "프로필 복장 이미지 생성 종료")

    async def _save_backup(run, job, pixels, prompt_id, positive, negative, state, saved_path, snapshot):
        save_backup = getattr(server.queue_manager, "save_backup", None)
        if not callable(save_backup):
            return
        try:
            await _write_backup(
                run, job, pixels, prompt_id, positive, negative, state,
                saved_path, snapshot, save_backup)
        except Exception as exc:
            traceback.print_exc()
            run["warnings"].append(
                f"{job['plan']['profile_label']}: 백업을 저장하지 못했습니다: {exc}")

    async def _write_backup(
            run, job, pixels, prompt_id, positive, negative, state,
            saved_path, snapshot, save_backup):
        plan = job["plan"]
        positive = str(state.get("positive") or positive)
        negative = str(state.get("negative") or negative)
        conversion_info = deepcopy(state.get("conversion_info") or {})
        conversion_info["emotion_asset"] = {
            "anchor": "",
            "character": plan["character"],
            "profile_id": plan["profile_id"],
            "outfit": plan["outfit"],
            "outfit_folder": plan["outfit_folder"],
            "outfit_tags": list(plan["outfit_tags"]),
            "emotion_id": "",
            "emotion_name": plan["emotion_name"],
            "rating": plan["rating"],
            "target_dir": os.path.dirname(saved_path),
            "saved_path": saved_path,
        }
        provider = str(state.get("provider") or snapshot.get("provider") or "comfy")
        await save_backup(
            pixels, prompt_id, positive, negative,
            generation_time=state.get("generation_time"),
            bot_name=run["bot_name"],
            gen_method=str(state.get("gen_method") or "감정 에셋"),
            workshop="emotion",
            provider=provider,
            provider_mode=str(state.get("provider_mode") or provider),
            prompt_provider=str(state.get("prompt_provider") or provider),
            generation_params=state.get("generation_params"),
            original_workflow_snapshot=state.get("original_workflow"),
            api_workflow_snapshot=state.get("api_workflow"),
            conversion_info_snapshot=conversion_info,
            submitted_prompt_snapshot=state.get("submitted_prompt"),
            illustration_visual_states=state.get("illustration_visual_states"),
            execution_target=str(state.get("execution_target") or ""),
        )

    async def _cancel_jobs(jobs):
        for job in jobs:
            queue_item = job.get("queue_item")
            if queue_item is None or queue_item.completion_future.done():
                continue
            try:
                await server.queue_manager.cancel_item(queue_item.id)
            except Exception:
                traceback.print_exc()

    async def collect(run, jobs, snapshot):
        token = storage.set_bot_scope("emotion")
        try:
            for job in jobs:
                plan = job["plan"]
                run["current"] = f"{plan['character']} · {plan['profile_label']}"
                try:
                    await _finish_job(run, job, snapshot)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    traceback.print_exc()
                    run["errors"].append(
                        f"{plan['character']} · {plan['profile_label']}: {exc}")
                finally:
                    run["done"] += 1
            if run["created"]:
                parts = [f"복장 이미지 {len(run['created'])}개 생성"]
                if run["cancelled"]:
                    parts.append(f"취소 {len(run['cancelled'])}개")
                run["status"] = "completed"
                run["message"] = " · ".join(parts) + " 완료"
                if run["errors"]:
                    run["error"] = f"{len(run['errors'])}개 실패: {run['errors'][0]}"
            elif run["errors"]:
                run["status"] = "failed"
                run["error"] = f"프로필 복장 이미지 생성에 실패했습니다: {run['errors'][0]}"
            elif run["cancelled"]:
                run["status"] = "cancelled"
                run["message"] = f"복장 이미지 {len(run['cancelled'])}개 생성을 취소했습니다."
            else:
                run["status"] = "completed"
                run["message"] = "생성할 프로필이 없습니다."
        except asyncio.CancelledError:
            await _cancel_jobs(jobs)
            run["status"] = "cancelled"
            run["message"] = ""
            run["error"] = "프로필 복장 이미지 생성이 중단되었습니다."
        except Exception as exc:
            await _cancel_jobs(jobs)
            traceback.print_exc()
            run["status"] = "failed"
            run["message"] = ""
            run["error"] = f"프로필 복장 이미지 생성에 실패했습니다: {exc}"
        finally:
            for job in jobs:
                server._discard_deferred_illustration_prompt(
                    job["prompt_id"], "프로필 복장 이미지 생성 종료")
            run["current"] = ""
            tasks.pop(run["bot_name"], None)
            storage.reset_bot_scope(token)

    async def handle(request):
        if request.method == "GET":
            bot_name = request.query.get("bot_name", "").strip()
            try:
                if not bot_name:
                    raise ValueError("봇을 선택하세요.")
                run = runs.get(bot_name)
                # Polling an active run must not rescan every profile folder.
                if run and run.get("status") in ("generating", "preparing"):
                    return web.json_response({"plans": run.get("plans") or [], "run": run})
                snapshot = server._capture_illustration_runtime_snapshot(
                    {**server.app_config, "emotion_bot_selected": bot_name},
                    workshop="emotion",
                )
                plans = await asyncio.to_thread(
                    _profile_outfit_plans, server, bot_name,
                    snapshot["illustration_context_toggles"])
                return web.json_response({
                    "plans": [_public_outfit_plan(plan) for plan in plans],
                    "run": run,
                })
            except (ValueError, OSError, KeyError, TypeError) as exc:
                return web.json_response({"error": str(exc)}, status=400)
        try:
            if storage.bot_scope() != "emotion":
                raise ValueError("감정 삽화 공방에서만 프로필 복장 이미지를 생성할 수 있습니다.")
            body = await request.json()
            if not isinstance(body, dict) or not isinstance(body.get("bot_name"), str) or not body["bot_name"].strip():
                raise ValueError("프로필 복장 이미지를 생성할 봇을 선택하세요.")
            bot_name = body["bot_name"].strip()
            if bot_name in tasks:
                return web.json_response(
                    {"run": runs[bot_name], "plans": []}, status=202)
            action = str(body.get("action") or "").strip().lower()
            if action:
                if action != "cancel":
                    raise ValueError("지원하지 않는 프로필 복장 작업입니다.")
                run = runs.get(bot_name)
                if run is None:
                    raise ValueError("취소할 프로필 복장 생성 작업이 없습니다.")
                run["cancel_requested"] = True
                batch_id = str(run.get("batch_id") or "")
                cancelled = 0
                cancel_batch = getattr(server.queue_manager, "cancel_batch", None)
                if batch_id and callable(cancel_batch):
                    cancelled = await cancel_batch(batch_id)
                return web.json_response(
                    {"run": run, "cancelled": int(cancelled or 0)}, status=202)
            # 삽화/감정 봇 이름이 같아도 emotion_bot_selected + workshop으로
            # 현재 감정 봇의 그리기 설정(프롬프트 프리셋 포함)을 고정한다.
            snapshot = server._capture_illustration_runtime_snapshot(
                {**server.app_config, "emotion_bot_selected": bot_name},
                workshop="emotion",
            )
            # 프로필별 복장 이미지는 에셋 생성 모드(감정 삽화 공방의 일반 삽화
            # 설정)를 읽는다. 새 복장 커버를 만드는 캐릭터 자동 생성 모드와
            # 분리해, 두 흐름이 서로의 크기·레퍼런스·디테일러 설정을 덮지 않게 한다.
            snapshot["generation_mode"] = "illustration"
            toggles = snapshot["illustration_context_toggles"]
            plans = await asyncio.to_thread(
                _profile_outfit_plans, server, bot_name, toggles)
            public_plans = [_public_outfit_plan(plan) for plan in plans]
            pending = [plan for plan in plans if plan["action"] == "generate"]
            prompts = pipeline.load_prompt_files("emotion")
            jobs, specs, failures = [], [], []
            for index, plan in enumerate(pending, start=1):
                try:
                    job = _profile_outfit_prompt_job(
                        server, plan, index, prompts, toggles, snapshot)
                    jobs.append(job)
                    specs.append({
                        "type": "illustration",
                        "label": f"프로필 복장 · {plan['character']} · {plan['profile_label']}",
                        "batch_label": f"프로필별 복장 이미지 생성 · {bot_name}",
                        "params": job["params"],
                    })
                except Exception as exc:
                    traceback.print_exc()
                    failures.append(
                        f"{plan['character']} · {plan['profile_label']}: {exc}")
            try:
                created = await server.queue_manager.add_items_batch(
                    specs, priority=0) if specs else []
            except Exception as exc:
                for job in jobs:
                    server._discard_deferred_illustration_prompt(
                        job["prompt_id"], "프로필 복장 생성 등록 실패")
                raise ValueError(
                    f"작업 큐에 프로필 복장 생성을 등록하지 못했습니다: {exc}") from exc
            for job, queue_item in zip(jobs, created):
                job["queue_item"] = queue_item
            run = {
                "bot_name": bot_name, "id": uuid.uuid4().hex,
                "status": "generating" if jobs else ("failed" if failures else "completed"),
                "message": (
                    f"프로필 복장 이미지 {len(pending)}개 생성 중…"
                    if jobs else ("" if failures else "생성할 프로필이 없습니다.")),
                "error": failures[0] if failures and not jobs else "",
                "total": len(pending), "done": len(failures),
                "current": "", "created": [], "cancelled": [],
                "warnings": [plan["warning"] for plan in plans if plan.get("warning")],
                "errors": list(failures), "cancel_requested": False,
                "batch_id": str(getattr(created[0], "batch_id", "") or "") if created else "",
                "plans": public_plans,
            }
            runs[bot_name] = run
            if jobs:
                tasks[bot_name] = asyncio.create_task(
                    collect(run, jobs, snapshot))
            return web.json_response({"run": run, "plans": public_plans}, status=202)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            return web.json_response({"error": str(exc)}, status=400)

    async def cleanup(_app):
        pending_tasks = list(tasks.values())
        for task in pending_tasks:
            task.cancel()
        await asyncio.gather(*pending_tasks, return_exceptions=True)

    server.app.router.add_get("/api/bot_mode/generate_profile_outfits", handle)
    server.app.router.add_post("/api/bot_mode/generate_profile_outfits", handle)
    server.app.on_cleanup.append(cleanup)
