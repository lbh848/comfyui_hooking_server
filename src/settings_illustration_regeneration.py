"""Regenerate a generated bot image, then refresh its normal reference assets."""
from __future__ import annotations

import asyncio
from copy import deepcopy
from importlib import import_module
import io
import json
import os
from pathlib import Path
import time
import traceback
import uuid

from aiohttp import web
from PIL import Image, PngImagePlugin

from modes import illustration_context_pipeline as pipeline
from modes.illustration_npc import _character_dir, character_image_descriptor, is_generated_image, save_tracked_face
from modes.visual_profiles import effective_character_cards, store_visual_cards
from settings_illustration_prepare import _call
import workflow_profiles

storage = import_module("modes.bot_mode")


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
            descriptor = character_image_descriptor(root)
            toggles = {**snapshot["illustration_context_toggles"], "positive_note": "", "negative_note": ""}
            positive, negative = pipeline.build_raw_prompt(descriptor, "", pipeline.load_prompt_files(), toggles)
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
            params = {"prompt_id": prompt_id, "prompt_data": prompt_data, "provider": provider,
                      "illustration_flow_skip": True,
                      "raw_body": {"illustration_provider": provider, "illustration_prompt_format": prompt_format,
                                   "illustration_defer_postprocess": True}}
            if root.get("npc_generated"):
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
            snapshot = server._capture_illustration_runtime_snapshot({**server.app_config, "bot_selected": body["bot_name"]})
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
