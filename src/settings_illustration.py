"""Local illustration gateway sharing the existing generation/LoRA worker."""
from __future__ import annotations

import asyncio
import json
import re
import secrets

import aiohttp

from aiohttp import web

API_PREFIX = "/api/illust"

# Keep the worker's public surface explicit. Never expose /api/config or arbitrary
# server routes (credentials, filesystem actions, and other applications).
READ_PATHS = {
    "bot_mode/regenerate_image",
    "backups", "backups/filters", "reschedule", "illustration/config", "asset_mode/tags",
    "illustration_context/bridge/bots", "illustration_context/prompts",
    "illustration_context/toggles", "illustration_context/prompt-structure",
    "illustration_flow", "restore_manual/characters", "queue/status",
    "bot_mode/bots", "bot_mode/distribution", "bot_mode/images", "bot_mode/character_cards",
    "bot_mode/illust_settings", "illustration/generation-presets", "bot_mode/system_prompt", "bot_mode/asset_output_instruction", "bot_mode/positive_rules",
    "frontend_ws", "bot_mode/check_patch_files",
    "bot_mode/data_preparation", "bot_mode/patch_settings", "bot_mode/rep_preview", "bot_mode/utility_preview", "bot_mode/tag_filter_profiles",
    "illustration_context/chat-histories", "illustration_context/chat-histories/settings",
    "bot_mode/illust_logs", "bot_mode/lb_extra", "bot_mode/lb_extra_refine_prompt", "bot_mode/auto_face_tag_prompt",
    "bot_mode/emotion_characters", "bot_mode/emotion_profiles", "bot_mode/emotion_outfits",
    "bot_mode/emotion_emotions", "bot_mode/emotion_assets",
    "bot_mode/importable_bot_chars", "bot_mode/asset_chars_with_rep", "bot_mode/asset_character_rep_images",
    "bot_mode/word_replacements", "illustration/thinking-connections", "illustration/extra",
    "bot_mode/postprocess_vn", "bot_mode/postprocess_bubble", "bot_mode/postprocess_subtitle",
    "postprocess/settings", "postprocess/fonts", "postprocess/face_devices", "illustration/workflow", "illustration/sampling-options", "illustration/presets",
    "llm_edit_prompt_template",
}
WRITE_PATHS = {
    "bot_mode/regenerate_image",
    "illustration/config", "illustration_context/bridge/bots",
    "illustration_context/prompts", "illustration_context/toggles",
    "illustration_context/prompt-structure", "illustration_context/execution-preview",
    "regenerate", "reschedule", "illustration_auto_feedback", "restore_manual_draw", "illustration_flow/cancel",
    "reschedule_with_modified_prompt", "llm_edit_prompt",
    "illustration/prepare",
    "bot_mode/action", "bot_mode/distribution", "bot_mode/upload", "bot_mode/upload_preflight", "bot_mode/delete_image",
    "bot_mode/character_cards", "bot_mode/update_illust_settings", "illustration/generation-presets",
    "bot_mode/character_cards/suggest_metadata", "bot_mode/character_cards/suggest_metadata/cancel", "bot_mode/character_cards/apply_metadata",
    "bot_mode/system_prompt", "bot_mode/asset_output_instruction", "bot_mode/positive_rules",
    "illustration_context/chat-histories/settings", "bot_mode/lb_extra",
    "bot_mode/lb_extra_refine", "bot_mode/lb_extra_refine_prompt",
    "bot_mode/auto_group_prompt", "bot_mode/auto_classify_face_tags", "bot_mode/auto_face_tag_prompt",
    "bot_mode/data_preparation",
    "bot_mode/import_asset_chars", "bot_mode/import_bot_chars",
    "bot_mode/patch_settings", "bot_mode/tag_filter_profile_save", "bot_mode/tag_filter_profile_delete", "bot_mode/tag_filter_preview",
    "bot_mode/word_replacements", "illustration/thinking-check",
    "bot_mode/emotion_profiles", "bot_mode/emotion_outfits", "bot_mode/emotion_emotions",
    "bot_mode/emotion_assets", "bot_mode/emotion_upload",
    "bot_mode/postprocess_vn", "bot_mode/postprocess_bubble", "bot_mode/postprocess_subtitle",
    "postprocess/settings", "postprocess/preview", "postprocess/font/upload", "postprocess/font/delete", "illustration/workflow", "illustration/presets",
    "llm_edit_prompt_template",
}
MEDIA_PATTERN = r"/api/(?:backup_(?:image|poster|raw)/[^/]+|bot_mode/image/[^/]+/[^/]+/[^/]+|bot_mode/emotion_image)"


def illustration_media_path(path: str) -> bool:
    return bool(re.fullmatch(MEDIA_PATTERN, path))


def allowed_illustration_path(path: str, method: str) -> bool:
    if not path.startswith("/api/"):
        return False
    name = path[5:]
    if name == "bot_mode/system_prompt_presets":
        return method == "DELETE"
    if re.fullmatch(r"illustration_context/chat-histories/[A-Za-z0-9_-]+", name):
        return method in {"GET", "HEAD", "POST", "DELETE"} and (name.rsplit("/", 1)[1] != "settings" or method != "DELETE")
    if method in {"GET", "HEAD"}:
        return name in READ_PATHS or illustration_media_path(path) or bool(
            re.fullmatch(r"(?:backup_prompt/[^/]+|backups/[^/]+/llm_trace|illustration_auto_feedback/[A-Za-z0-9_-]+)", name)
        )
    return method == "POST" and (name in WRITE_PATHS or bool(
        re.fullmatch(r"(?:backup_delete/[^/]+|backups/[^/]+/llm_trace|illustration_auto_feedback/[A-Za-z0-9_-]+/cancel)", name)
    ))


async def illustration_sampling_options(server, task_key="illustration") -> dict:
    """Read the illustration runtime's actual choices, including custom samplers."""
    try:
        port = server.resolve_comfy_port(task_key)
        url = f"http://{server.REAL_COMFY_HOST}:{port}/object_info/KSampler"
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3)) as session:
            async with session.get(url) as response:
                response.raise_for_status()
                info = await response.json()
        required = info["KSampler"]["input"]["required"]
        options = {}
        for field in ("sampler_name", "scheduler"):
            choices = required[field][0]
            if not isinstance(choices, list) or not choices or not all(isinstance(item, str) and item for item in choices):
                raise ValueError("Invalid KSampler choices")
            options[field] = list(dict.fromkeys(choices))
        return {"source": "comfyui", **options}
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError, ValueError, KeyError, TypeError, IndexError):
        # The UI bundles core choices so editing also works with ComfyUI stopped
        # or an illustration task assigned to an on-demand remote runtime.
        return {"source": "default"}


def register_illustration(app: web.Application, runtime, schema: str) -> None:
    connections = set()

    async def events(request: web.Request) -> web.StreamResponse:
        # Browser WebSockets cannot set fetch headers; the app's loopback
        # Host/Origin middleware remains mandatory for this read-only stream.
        if request.query.get("schema") != schema:
            raise web.HTTPConflict(text="삽화 화면을 새로고침해 주세요.")
        try:
            await runtime.start()
            upstream = await runtime.session.ws_connect(
                runtime.url + "/api/frontend_ws", headers={"X-Internal-Lora": runtime.token}, heartbeat=30,
            )
        except (RuntimeError, OSError, aiohttp.ClientError, asyncio.TimeoutError) as exc:
            return web.json_response({"error": str(exc)}, status=503)
        socket = web.WebSocketResponse(heartbeat=30)
        await socket.prepare(request)
        connections.add(socket)

        async def receive():
            async for message in upstream:
                if message.type == aiohttp.WSMsgType.TEXT:
                    event = json.loads(message.data)
                    if event.get("type") == "ping":
                        await upstream.send_json({"type": "pong"})
                    elif event.get("type") in {"queue_updated", "queue_progress", "backup_created", "backup_filter_ready", "bot_selected_changed", "illustration_warning", "reschedule_changed", "reschedule_used", "asset_generation_completed", "qwen_edit_completed"}:
                        await socket.send_json(event)

        async def disconnected():
            async for _ in socket:
                pass

        tasks = [asyncio.create_task(receive()), asyncio.create_task(disconnected())]
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            connections.discard(socket)
            await upstream.close()
            await socket.close()
        return socket

    async def shutdown(_app):
        for socket in list(connections):
            await socket.close(code=1001, message=b"Server shutdown")

    async def gateway(request: web.Request) -> web.StreamResponse:
        path = "/api/" + request.match_info["path"]
        if path == "/api/frontend_ws" or not allowed_illustration_path(path, request.method):
            raise web.HTTPNotFound()
        if not (request.method == "GET" and illustration_media_path(path)):
            if request.headers.get("X-Comfy-Settings") != "1":
                raise web.HTTPForbidden()
            if request.headers.get("X-Comfy-Settings-Schema") != schema:
                return web.json_response({"error": "삽화 화면을 새로고침해 주세요."}, status=409)
        return await runtime.request(request, illustration=True)

    app.router.add_get(API_PREFIX + "/events", events)
    app.on_shutdown.append(shutdown)
    app.router.add_get(API_PREFIX + "/{path:.+}", gateway)
    app.router.add_post(API_PREFIX + "/{path:.+}", gateway)
    app.router.add_delete(API_PREFIX + "/{path:.+}", gateway)


def register_illustration_config(server, *, model_presets=None) -> None:
    """Reuse the normal configuration writer, returning only illustration fields."""
    import workflow_profiles

    keys = {
        "illustration_workflow_type", "illustration_workflow_source_paths",
        "postprocess_enabled", "chansub_workflow_type", "backup_max_count",
    }

    async def config(request: web.Request) -> web.Response:
        scope = str(request.query.get("scope") or "").strip().lower()
        emotion_scope = scope == "emotion"
        if request.method == "POST" and emotion_scope:
            body = await request.json()
            if not isinstance(body, dict) or set(body) - {"postprocess_enabled"}:
                return web.json_response({"error": "지원하지 않는 감정 삽화 설정입니다."}, status=400)
            if not isinstance(body.get("postprocess_enabled"), bool):
                return web.json_response({"error": "감정 삽화 후처리 설정 값을 확인하세요."}, status=400)
            server.app_config["emotion_postprocess_enabled"] = body["postprocess_enabled"]
            server.save_config(server.app_config)
            # Return the same key set as GET so the autosave keeps a complete
            # baseline and never resends shared illustration settings.
            config_values = {
                key: server.app_config.get(key, server.DEFAULT_CONFIG.get(key)) for key in keys
            }
            config_values["postprocess_enabled"] = bool(
                server.app_config["emotion_postprocess_enabled"]
            )
            return web.json_response(
                {"config": config_values},
                headers={"Cache-Control": "no-store"},
            )
        if request.method == "POST":
            body = await request.json()
            if not isinstance(body, dict) or set(body) - keys:
                return web.json_response({"error": "지원하지 않는 삽화 설정입니다."}, status=400)
            for key, value in body.items():
                if key == "backup_max_count":
                    if type(value) is not int or not 1 <= value <= 10000:
                        return web.json_response({"error": "백업 보관 개수는 1~10,000 사이의 정수여야 합니다."}, status=400)
                    continue
                valid = (
                    key == "illustration_workflow_type" and isinstance(value, str)
                    and value in workflow_profiles.ILLUSTRATION_WORKFLOW_TYPES
                    or key == "postprocess_enabled" and isinstance(value, bool)
                    or key == "chansub_workflow_type" and isinstance(value, str) and value in {"anima", "sdxl"}
                    or key == "illustration_workflow_source_paths" and isinstance(value, dict)
                    and set(value) <= set(workflow_profiles.ILLUSTRATION_LOCAL_PROFILES)
                    and all(isinstance(item, str) for item in value.values())
                )
                if not valid:
                    return web.json_response({"error": "삽화 설정 값을 확인하세요."}, status=400)
            response = await server.handle_api_config(request)
            if response.status >= 400:
                return response
        config_values = {
            key: server.app_config.get(key, server.DEFAULT_CONFIG.get(key)) for key in keys
        }
        if emotion_scope:
            config_values["postprocess_enabled"] = bool(
                server.app_config.get(
                    "emotion_postprocess_enabled",
                    server.app_config.get("postprocess_enabled", False),
                )
            )
        return web.json_response({
            "config": config_values,
            "workflow_types": workflow_profiles.ILLUSTRATION_PROFILE_LABELS,
            "local_profiles": workflow_profiles.ILLUSTRATION_LOCAL_PROFILES,
            "capabilities": workflow_profiles.illustration_capabilities(
                server.app_config.get("illustration_workflow_type"),
                server.app_config.get("chansub_workflow_type", "anima"),
            ),
        }, headers={"Cache-Control": "no-store"})

    server.app.router.add_get("/api/illustration/config", config)
    server.app.router.add_post("/api/illustration/config", config)

    async def sampling_options(request: web.Request) -> web.Response:
        return web.json_response(await illustration_sampling_options(server), headers={"Cache-Control": "no-store"})

    server.app.router.add_get("/api/illustration/sampling-options", sampling_options)

    from illustration_generation_presets import register_generation_presets
    register_generation_presets(server.app, server.CONFIG_FILE)

    from illustration_model_presets import IllustrationModelPresets, apply_active_model_preset, register_model_preset_runtime
    presets = model_presets or IllustrationModelPresets(server.CONFIG_FILE)
    register_model_preset_runtime(server, "illustration", presets,
        lambda: apply_active_model_preset(server.app_config, server.CONFIG_FILE, presets.builtin_file),
        workshop="illustration")

    async def prepare(request: web.Request) -> web.Response:
        body = await request.json()
        if not isinstance(body, dict) or set(body) - {"bot_name", "char_name"} or not all(isinstance(value, str) for value in body.values()):
            return web.json_response({"error": "캐릭터를 선택하세요."}, status=400)
        bot_name, char_name = body.get("bot_name", ""), body.get("char_name", "")
        from modes.bot_mode import get_bot_visual_targets
        if not bot_name or not char_name or not get_bot_visual_targets(bot_name, char_name, require_rep_images=True):
            return web.json_response({"error": "캐릭터 카드에 대표 이미지를 저장한 뒤 데이터 패치를 실행하세요."}, status=400)
        response = await server.data_patcher.handle_data_patch(request)
        if response.status >= 400:
            return response
        result = json.loads(response.body)
        items = []
        for target in result.get("visual_targets", []):
            item = await server.queue_manager.add_item("data_patch_utility", f"데이터 패치 · {target['character']}", {
                "bot_name": bot_name, "char_name": target["character"],
                "visual_card_id": target["visual_card_id"], "visual_card_label": target["visual_card_label"],
                "visual_card_index": target["visual_card_index"],
            })
            items.append(item.id)
        return web.json_response({"queued": True, "item_ids": items, "message": result["message"]}, status=202)

    server.app.router.add_post("/api/illustration/prepare", prepare)

    thinking_snapshots = {}

    def thinking_signature(slot, config):
        from llm_settings import SLOT_DEFAULTS

        suffix = "" if slot == "llm1" else slot[3:]
        fields = (*SLOT_DEFAULTS, "service", "url", "api_key", "max_concurrency")
        snapshot = json.dumps({field: config.get(f"llm_{field}{suffix}") for field in fields}, sort_keys=True)
        previous = thinking_snapshots.get(slot)
        if previous is None or previous[0] != snapshot:
            # An opaque revision detects credential changes without exposing them
            # or publishing a credential-derived hash to the browser.
            thinking_snapshots[slot] = (snapshot, secrets.token_hex(16))
        return thinking_snapshots[slot][1]

    async def thinking_connections(request: web.Request) -> web.Response:
        from modes.illustration_quality_inspection import TASK_KEY, CORRECTION_TASK_KEY, SELECTION_TASK_KEY

        routes, slots = {}, {}
        config = server.llm_service.get_config()
        tasks = set(server.illustration_context_pipeline._CALL_TASK_KEYS.values())
        tasks.update((TASK_KEY, CORRECTION_TASK_KEY, SELECTION_TASK_KEY))
        for task in sorted(tasks):
            try:
                primary, fallback = server.llm_service._routing_for(task, config=config)
            except ValueError:
                routes[task] = {"primary": None, "fallback": None}
                continue
            routes[task] = {"primary": primary, "fallback": fallback}
            for slot in (primary, fallback):
                if not slot or slot in slots:
                    continue
                suffix = "" if slot == "llm1" else slot[3:]
                slots[slot] = {"id": slot, "model": config.get(f"llm_model{suffix}", ""),
                               "alias": config.get(f"llm_alias{suffix}", ""),
                               "service": config.get(f"llm_service{suffix}", ""),
                               "signature": thinking_signature(slot, config)}
        return web.json_response({"routes": routes, "slots": slots})

    async def thinking_check(request: web.Request) -> web.StreamResponse:
        body = await request.json()
        if (not isinstance(body, dict) or set(body) - {"target", "thinking_check", "signature"}
                or body.get("thinking_check") is not True
                or body.get("target") not in server.llm_service.LLM_SLOT_IDS):
            return web.json_response({"error": "검사할 LLM 연결을 선택하세요."}, status=400)
        if "signature" in body and body["signature"] != thinking_signature(body["target"], server.llm_service.get_config()):
            return web.json_response({"error": "LLM 연결 설정이 변경되었습니다. 흐름 새로고침 후 다시 확인하세요."}, status=409)
        return await server.handle_api_llm_test_stream(request)

    server.app.router.add_get("/api/illustration/thinking-connections", thinking_connections)
    server.app.router.add_post("/api/illustration/thinking-check", thinking_check)

    async def extra(request: web.Request) -> web.Response:
        bot_name = request.query.get("bot_name", "").strip()
        collected = server._collect_lb_extra_impl(bot_name)
        if collected is None:
            return web.json_response({"error": "봇을 선택하고 캐릭터를 등록하세요."}, status=404)
        return web.json_response({"bot_name": bot_name, "characters": collected["visual_profiles"]})

    server.app.router.add_get("/api/illustration/extra", extra)
