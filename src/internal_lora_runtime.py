# /// script
# requires-python = ">=3.12.11,<3.13"
# dependencies = [
#   "aiohttp", "httpx", "pydantic", "pillow", "pillow-avif-plugin", "numpy",
#   "packaging", "argon2-cffi", "cryptography", "google-auth", "requests",
#   "psutil", "piexif", "pyyaml", "modal>=1.5.3,<2", "paramiko",
#   "onnxruntime", "huggingface-hub", "opencv-python-headless",
# ]
# ///
"""Run the existing illustration and LoRA handlers on a shared local queue."""
from __future__ import annotations

import argparse
import asyncio
import os
import secrets

from aiohttp import web

from settings_internal_lora import allowed_path
from settings_illustration import allowed_illustration_path, register_illustration_config
from settings_assets import allowed_asset_path, register_asset_runtime
from settings_illustration_prepare import register_data_preparation
from external_connection import allowed_external_path, allowed_external_control, register_external_connection_control


CONFIG_KEYS = (
    "lora_load_path", "bot_lora_load_path", "style_lora_load_path",
    "lora_training_workflow_source_paths", "style_lora_training_workflow_source_paths",
    "tag_analysis_workflow_source_path", "asset_tag_analysis_workflow_source_path",
    "use_builtin_tagger", "comfy_input_dir",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    token = os.environ.pop("SOYA_INTERNAL_LORA_TOKEN", "")
    if not token:
        parser.error("This worker must be launched by the settings server.")

    import server

    def refresh_config():
        # Other settings pages write the same configuration while this worker lives.
        value = server.load_config()
        if value != server.app_config:
            server.app_config = value
            server.REAL_COMFY_PORT = int(value.get("comfyui_port", 8188))
            server.REAL_COMFY_ILLUST_PORT = value.get("comfyui_port_illustration")
            server.llm_service.update_config({key: value.get(key, default)
                for key, default in server.llm_service.get_config().items() if key in server.DEFAULT_CONFIG})

    @web.middleware
    async def internal_only(request, handler):
        if not secrets.compare_digest(request.headers.get("X-Internal-Lora", ""), token):
            raise web.HTTPForbidden()
        illustration = allowed_illustration_path(request.path, request.method)
        asset = allowed_asset_path(request.path, request.method)
        external = allowed_external_path(request.path, request.method)
        control = allowed_external_control(request.path, request.method)
        if not allowed_path(request.path) and not illustration and not asset and not external and not control:
            raise web.HTTPNotFound()
        if request.method == "POST" or illustration or asset or external:
            refresh_config()
        if request.method == "POST":
            server._load_llm_keys_into_config()
            if illustration or external:
                server._load_chansub_key()
        from importlib import import_module
        bot_mode_module = import_module("modes.bot_mode")
        scope_token = bot_mode_module.set_bot_scope(
            "emotion" if request.query.get("scope") == "emotion" else "illust"
        )
        try:
            return await handler(request)
        finally:
            bot_mode_module.reset_bot_scope(scope_token)

    async def status(request):
        return web.json_response({"ready": True})

    async def config(request):
        value = server.load_config()
        if request.method == "POST":
            body = await request.json()
            if not isinstance(body, dict) or set(body) - set(CONFIG_KEYS):
                raise web.HTTPBadRequest(text="Unsupported LoRA setting")
            for key, item in body.items():
                if key.endswith("source_paths"):
                    if not isinstance(item, dict) or set(item) - {"anima", "sdxl"} or not all(isinstance(v, str) for v in item.values()):
                        raise web.HTTPBadRequest(text="Invalid workflow paths")
                elif key == "use_builtin_tagger":
                    if not isinstance(item, bool):
                        raise web.HTTPBadRequest(text="Invalid tagger setting")
                elif not isinstance(item, str):
                    raise web.HTTPBadRequest(text="Invalid LoRA path")
            if body:
                response = await server.handle_api_config(request)
                if response.status >= 400:
                    return response
                value = server.app_config
        return web.json_response({"success": True, "data": {key: value.get(key, server.DEFAULT_CONFIG.get(key, "")) for key in CONFIG_KEYS}})

    async def startup(app):
        server._main_event_loop = asyncio.get_running_loop()
        server.init_queue_manager()
        server.llm_service.update_config({key: server.app_config.get(key, default)
            for key, default in server.llm_service.get_config().items() if key in server.DEFAULT_CONFIG})
        server._load_llm_keys_into_config()
        server._load_chansub_key()

    async def preview(request):
        from modes import lora_mode, bot_lora_mode
        category = request.query.get("category")
        character = request.query.get("character", "")
        entry = request.query.get("entry", "")
        if category == "asset":
            info = next((e for e in lora_mode.list_lora_entries(character) if e["name"] == entry), None)
            if info is None:
                return web.json_response({"success": False, "error": "프로젝트를 찾을 수 없습니다."}, status=404)
            cfg = info.get("training_config", {})
            images = lora_mode.list_training_images(character, entry)
            tests = lora_mode.list_test_images(character, entry)
            trigger = info.get("trigger", "")
            save_path = lora_mode.training_save_path(character, entry)
        elif category == "bot":
            bot, project = request.query.get("bot", ""), request.query.get("project", "")
            card = request.query.get("visual_card_id", "")
            detail = bot_lora_mode.get_project_data(bot, project)
            if not detail.get("success"):
                return web.json_response(detail, status=400)
            unit = next((c for c in detail["characters"] if c["name"] == character and c["visual_card_id"] == card), None)
            if unit is None:
                return web.json_response({"success": False, "error": "캐릭터 카드를 선택하세요."}, status=400)
            cfg = detail["training_config"]
            images = unit["training_images"]
            tests = unit["char_test_images"] or detail["test_images"]
            trigger = unit["trigger"]
            save_path = bot_lora_mode.training_save_path(bot, project, character, card, cfg)
        else:
            raise web.HTTPBadRequest()
        from lora_training_steps import prepare_training_images
        images, step = prepare_training_images(images, cfg)
        args = (images, trigger, cfg.get("profile", "anima"), step,
                cfg.get("il_rate", .0005), cfg.get("save_per_step", 50), cfg.get("multi_img_folder_name", "soya_lora"))
        kwargs = dict(lora_save_path=save_path, gen_w=cfg.get("gen_w", 1024), gen_h=cfg.get("gen_h", 1024),
                      upscale=cfg.get("upscale", False), resolution=cfg.get("resolution", 1024),
                      test_images=tests, save_after=cfg.get("save_after", 0), dim=cfg.get("dim", 32), alpha=cfg.get("alpha", 16))
        return web.json_response({"success": True, "positive": server._build_lora_training_text(*args, field="positive", **kwargs),
                                  "negative": server._build_lora_training_text(*args, field="negative", **kwargs)})

    server.app.middlewares.insert(0, internal_only)
    server.app.on_startup.clear()
    server.app.on_startup.append(startup)
    register_illustration_config(server)
    register_asset_runtime(server)
    register_data_preparation(server)
    from settings_illustration_regeneration import register_character_image_regeneration
    register_character_image_regeneration(server)
    register_external_connection_control(server)
    server.app.router.add_get("/api/internal-lora/status", status)
    server.app.router.add_get("/api/internal-lora/config", config)
    server.app.router.add_post("/api/internal-lora/config", config)
    server.app.router.add_get("/api/internal-lora/preview", preview)
    web.run_app(server.app, host="127.0.0.1", port=args.port, access_log=None)


if __name__ == "__main__":
    main()
