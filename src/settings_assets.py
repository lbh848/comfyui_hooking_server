"""Local asset workspace using the existing asset handlers and shared queue."""
from __future__ import annotations

import re
import json
from aiohttp import web

CONFIG_KEYS = {
    "asset_workflow_type", "asset_workflow_source_path",
    "anima_asset_workflow_source_path", "anima_only_asset_workflow_source_path",
    "moss_asset_workflow_source_path",
}
READ_PATHS = {
    "asset/character-maker", "asset/edit-options",
    "asset/presets", "asset/style-presets", "asset/character-presets", "asset/workflow", "asset/sampling-options",
    "asset/config", "asset_mode/status", "asset_mode/tags", "asset_mode/characters",
    "chain_presets", "queue/status", "asset_mode/qwen_edit/composite_items",
}
WRITE_PATHS = {
    "asset/character-maker", "asset/character-upload",
    "asset/presets", "asset/style-presets", "asset/character-presets", "asset/workflow", "asset/style-ref/upload", "asset/style-ref/import", "asset/preview",
    "asset/config", "asset/enqueue", "asset_mode/tags", "asset_mode/upload_reference",
    "asset_mode/compute_ref_hash", "asset_mode/set_representative", "asset_mode/delete_image", "asset_mode/delete_combination",
    "asset_mode/qwen_edit/enqueue", "asset_mode/qwen_edit/translate",
    "asset_mode/qwen_edit/composite_items", "asset_mode/qwen_edit/composite_items/background_remove",
    "asset_mode/name_mapping", "asset_mode/name_mapping/validate", "asset_mode/name_mapping/llm",
    "asset_mode/export", "asset/export", "asset/export/validate",
    "chain_presets/save", "chain_presets/delete", "chain_presets/manage",
}
MEDIA_PATTERN = r"/api/(?:asset/character-maker/image|asset/style-ref/image/[a-f0-9]{24}\.png|asset_mode/characters/[^/]+/outfits/[^/]+/expressions/[^/]+/images/[^/]+|asset_mode/qwen_edit/composite_items/[^/]+\.[pP][nN][gG])"


def allowed_asset_path(path: str, method: str) -> bool:
    if not path.startswith("/api/"):
        return False
    name = path[5:]
    if method in {"GET", "HEAD"}:
        return name in READ_PATHS or bool(re.fullmatch(MEDIA_PATTERN, path)) or bool(re.fullmatch(
            r"(?:asset_mode/characters/[^/]+/(?:gallery|outfits/[^/]+/expressions/[^/]+/images)|asset_mode/name_mapping/[^/]+|chain_presets/[^/]+)", name,
        ))
    if method == "DELETE":
        return bool(re.fullmatch(r"asset_mode/qwen_edit/composite_items/[^/]+\.[pP][nN][gG]", name))
    return method == "POST" and name in WRITE_PATHS


def register_assets(app: web.Application, runtime, schema: str) -> None:
    async def gateway(request: web.Request) -> web.StreamResponse:
        path = "/api/" + request.match_info["path"]
        media = request.method in {"GET", "HEAD"} and re.fullmatch(MEDIA_PATTERN, path)
        if not media:
            if request.headers.get("X-Comfy-Settings") != "1":
                raise web.HTTPForbidden()
            if request.headers.get("X-Comfy-Settings-Schema") != schema:
                return web.json_response({"error": "에셋 화면을 새로고침해 주세요."}, status=409)
        return await runtime.request(request, asset=True)

    app.router.add_route("GET", "/api/asset/{path:.+}", gateway)
    app.router.add_route("POST", "/api/asset/{path:.+}", gateway)
    app.router.add_route("DELETE", "/api/asset/{path:.+}", gateway)


def register_asset_runtime(server) -> None:
    from asset_model_presets import AssetModelPresets, apply_asset_preset
    from asset_style_presets import AssetStylePresets, register_style_preset_runtime
    from illustration_model_presets import register_model_preset_runtime
    presets = AssetModelPresets(server.CONFIG_FILE)
    styles = AssetStylePresets(server.CONFIG_FILE)
    register_model_preset_runtime(server, "asset", presets)
    register_style_preset_runtime(server)
    if getattr(server, "asset_mode", None) is not None:
        from asset_character_presets import register_character_preset_runtime
        from settings_asset_export import register_asset_export_runtime
        register_character_preset_runtime(server)
        register_asset_export_runtime(server)

    async def sampling_options(request: web.Request) -> web.Response:
        from settings_illustration import illustration_sampling_options
        return web.json_response(await illustration_sampling_options(server, "asset_generation"),
                                 headers={"Cache-Control": "no-store"})

    server.app.router.add_get("/api/asset/sampling-options", sampling_options)

    async def edit_options(request: web.Request) -> web.Response:
        mode = server.qwen_edit_mode
        edit_tool = mode.normalize_edit_tool(None)
        return web.json_response({"edit_tool": edit_tool, "label": mode.edit_tool_label(edit_tool),
                                  "defaults": mode.edit_defaults(edit_tool)},
                                 headers={"Cache-Control": "no-store"})

    server.app.router.add_get("/api/asset/edit-options", edit_options)

    async def config(request: web.Request) -> web.Response:
        if request.method == "POST":
            body = await request.json()
            if not isinstance(body, dict) or set(body) - CONFIG_KEYS or not all(isinstance(v, str) for v in body.values()):
                return web.json_response({"error": "에셋 워크플로우 설정을 확인하세요."}, status=400)
            response = await server.handle_api_config(request)
            if response.status >= 400:
                return response
        return web.json_response({key: server.app_config.get(key, server.DEFAULT_CONFIG.get(key, "")) for key in CONFIG_KEYS})

    def prepare_job(job, style_presets):
        from asset_json import (
            asset_document, build_asset_moss_document,
            prepare_asset_mask_document, prepare_asset_references,
        )
        if not isinstance(job.get("character"), str) or not job["character"].strip():
            raise ValueError("캐릭터를 선택하세요.")
        # Preset names only organize output files. The prompt already contains applied values.
        for key in ("appearance", "outfit", "expression"):
            value = job.get(key)
            if value is not None and not isinstance(value, str):
                raise ValueError("캐릭터 프리셋 이름을 확인하세요.")
            job[key] = value if value and value.strip() else "기본"
        identity = job.get("asset_style_preset_id")
        if not isinstance(identity, str) or not any(preset["id"] == identity for preset in style_presets):
            raise ValueError("사용할 그림체 프리셋을 선택하세요. 삭제된 프리셋은 다시 선택해야 합니다.")
        snapshot = presets.snapshot(server.app_config, job.get("asset_model_preset_id"))
        profile = snapshot["config"]["asset_workflow_type"]
        if profile != job["asset_workflow_type"]:
            raise ValueError("그림 모델이 변경되었습니다. 프리셋을 다시 선택하세요.")
        if not snapshot.get("workflows", {}).get(profile, {}).get("fields"):
            raise ValueError("그림 모델 설정에서 워크플로우를 선택하세요.")
        document = job.get("prompt_json")
        if not isinstance(document, dict) or not isinstance(document.get("fields"), dict) or not document["fields"]:
            raise ValueError("프롬프트 JSON의 fields를 확인하세요.")
        document = asset_document(json.dumps(document, ensure_ascii=False, allow_nan=False))
        if profile == "moss":
            from modes.moss_prompt_builder import MossPromptBuilder
            document = build_asset_moss_document(document, snapshot)
            job["snapref_enabled"] = False
            job["snapref_embedded"] = True
            job["prompt_json"] = document
            job["positive_prompt"] = MossPromptBuilder.dumps(document)
            job["negative_prompt"] = ""
            job.update(face_id_enabled=False, style_ref_enabled=False, pose_enabled=False, hrf_control_net=False)
            return snapshot
        document = apply_asset_preset(document, snapshot)
        document = prepare_asset_mask_document(document)
        if document["fields"].get("MULTI_CHAR", {}).get("layout"):
            references = job.get("regional_references", [])
            names = document["fields"]["MULTI_CHAR"]["char_name_list"]
            if not isinstance(references, list) or any(not isinstance(ref, dict) or ref.get("CHAR") not in names for ref in references):
                raise ValueError("영역별 참조 이미지를 다시 선택하세요.")
            if len({ref["CHAR"] for ref in references}) != len(references):
                raise ValueError("같은 영역의 참조 이미지가 중복되었습니다.")
            job["regional_references"] = references
            job["snapref_enabled"] = bool(references)
            document["fields"].setdefault("SNAPREF", {})["enabled"] = bool(references)
        document = prepare_asset_references(document, job)
        job["snapref_enabled"] = document["fields"]["SNAPREF"]["enabled"]
        # The queued JSON owns the images; later edits/deletions cannot change this job.
        job["snapref_embedded"] = True
        job["prompt_json"] = document
        job["positive_prompt"] = json.dumps(document, ensure_ascii=False, allow_nan=False)
        job["negative_prompt"] = ""
        job.update(face_id_enabled=False, style_ref_enabled=False, pose_enabled=False, hrf_control_net=False)
        return snapshot

    if getattr(server, "asset_mode", None) is not None:
        from asset_character_maker import register_character_maker_runtime
        register_character_maker_runtime(server, lambda job: prepare_job(job, styles.read()["presets"]))

    async def preview(request: web.Request) -> web.Response:
        try:
            job = await request.json()
            if not isinstance(job, dict):
                raise ValueError("프롬프트 JSON을 확인하세요.")
            prepare_job(job, styles.read()["presets"])
            return web.json_response(job, headers={"Cache-Control": "no-store"})
        except (ValueError, TypeError, KeyError, OSError) as exc:
            return web.json_response({"error": str(exc)}, status=400)

    async def enqueue(request: web.Request) -> web.Response:
        body = await request.json()
        jobs = body.get("jobs") if isinstance(body, dict) else None
        if not isinstance(jobs, list) or not jobs:
            return web.json_response({"error": "생성할 에셋이 없습니다."}, status=400)
        # Validate the complete submission before any job starts.
        try:
            style_presets = styles.read()["presets"]
        except (ValueError, TypeError, OSError) as exc:
            return web.json_response({"error": str(exc)}, status=400)
        snapshots = []
        for index, job in enumerate(jobs):
            if not isinstance(job, dict):
                return web.json_response({"error": f"슬롯 {index + 1}: 생성할 에셋을 확인하세요."}, status=400)
            if job.get("asset_workflow_type") not in {"anima_only", "anima_ilxl", "moss"}:
                return web.json_response({"error": f"슬롯 {index + 1}: ANIMA, ANIMA+ILXL 또는 MOSS 워크플로우를 선택하세요."}, status=400)
            try:
                snapshot = prepare_job(job, style_presets)
                snapshots.append(snapshot)
            except (ValueError, TypeError, KeyError, OSError) as exc:
                return web.json_response({"error": f"슬롯 {index + 1}: {exc}"}, status=400)
        label = body.get("label") or "에셋 생성"
        items = await server.queue_manager.add_items_batch([
            {"type": "asset_generation", "label": f"{job['character']}/{job['outfit']}/{job['expression']}",
             "batch_label": label, "params": {"body": job, "asset_model_preset": snapshot}}
            for job, snapshot in zip(jobs, snapshots)
        ])
        return web.json_response({"success": True, "count": len(items), "batch_id": items[0].batch_id})

    async def character_upload(request: web.Request) -> web.Response:
        from io import BytesIO
        from PIL import Image
        from asset_json import CHARACTER_UPLOAD_EXPRESSION, CHARACTER_UPLOAD_OUTFIT
        try:
            if getattr(server, "asset_mode", None) is None:
                raise ValueError("에셋 모드가 준비되지 않았습니다.")
            asset_tool = getattr(server, "asset_tool", None)
            if asset_tool is None or (not asset_tool.use_builtin_tagger and not asset_tool.workflow_source_path):
                raise ValueError("태그 분석 워크플로우 경로가 설정되지 않았습니다. 워크플로우 관리에서 태그 분석 워크플로우를 확인하세요.")
            reader = await request.multipart()
            metadata_raw = ""
            images = []
            async for part in reader:
                if part.name == "metadata":
                    metadata_raw = (await part.read()).decode("utf-8")
                elif part.name == "image":
                    data = await part.read()
                    if data:
                        images.append((part.filename or "upload.png", data))
            if not images:
                raise ValueError("업로드할 이미지 파일을 선택하세요.")
            if not metadata_raw:
                raise ValueError("업로드 설정 정보를 확인하세요.")
            metadata = json.loads(metadata_raw)
            if not isinstance(metadata, dict) or not isinstance(metadata.get("job"), dict):
                raise ValueError("업로드 프롬프트 설정을 확인하세요.")
            job = dict(metadata["job"])
            if job.get("asset_workflow_type") not in {"anima_only", "anima_ilxl", "moss"}:
                raise ValueError("ANIMA, ANIMA+ILXL 또는 MOSS 워크플로우를 선택하세요.")
            character = job.get("character")
            if not isinstance(character, str) or not character.strip():
                raise ValueError("캐릭터를 선택하세요.")
            if character not in server.asset_mode.character_presets.catalog()["characters"]:
                raise ValueError("캐릭터를 찾을 수 없습니다. 다시 선택하세요.")
            raw_background = metadata.get("background_tags", "")
            if isinstance(raw_background, list):
                background_tags = ", ".join(tag.strip() for tag in raw_background if isinstance(tag, str) and tag.strip())
            elif isinstance(raw_background, str):
                background_tags = raw_background.strip()
            else:
                raise ValueError("배경 태그 형식을 확인하세요.")
            # Ignore configured outfit/expression and store uploaded character images under 참고복장/참고표정.
            job["outfit"] = CHARACTER_UPLOAD_OUTFIT
            job["expression"] = CHARACTER_UPLOAD_EXPRESSION
            job["appearance"] = ""
            job["snapref_enabled"] = False
            job["snapref_images"] = []
            job["snapref_face_images"] = []
            job["snapref_secondary_images"] = []
            if isinstance(job.get("prompt_json"), dict) and isinstance(job["prompt_json"].get("fields"), dict):
                snapref = job["prompt_json"]["fields"].get("SNAPREF")
                if isinstance(snapref, dict):
                    snapref["enabled"] = False
                    snapref["references"] = []
                    snapref.pop("dir", None)
            prepare_job(job, styles.read()["presets"])
            for _, image_data in images:
                try:
                    with Image.open(BytesIO(image_data)) as opened:
                        if opened.format not in {"PNG", "JPEG", "WEBP"}:
                            raise ValueError("PNG, JPEG, WebP 이미지 파일을 선택하세요.")
                        opened.load()
                except ValueError:
                    raise
                except Exception as exc:
                    raise ValueError("이미지 파일을 읽을 수 없습니다.") from exc
            saved = [
                server.asset_mode.save_uploaded_character_image(
                    character,
                    CHARACTER_UPLOAD_OUTFIT,
                    CHARACTER_UPLOAD_EXPRESSION,
                    image_data,
                    job["positive_prompt"],
                    job.get("negative_prompt", ""),
                )
                for _, image_data in images
            ]
            if server.asset_mode.notify_frontend_func:
                for entry in saved:
                    await server.asset_mode.notify_frontend_func("asset_generation_completed", {
                        "status": "success",
                        "character": character,
                        "outfit": CHARACTER_UPLOAD_OUTFIT,
                        "expression": CHARACTER_UPLOAD_EXPRESSION,
                        "filename": entry["filename"],
                        "storage_group": "",
                        "storage_outfit": CHARACTER_UPLOAD_OUTFIT,
                    })
            batch_label = f"캐릭터 업로드 프롬프트 태깅 · {character} ({len(saved)}장)"
            items_spec = [
                {
                    "type": "tag_analysis",
                    "label": f"캐릭터 업로드 태깅: {character}/{CHARACTER_UPLOAD_OUTFIT}/{CHARACTER_UPLOAD_EXPRESSION}/{entry['filename']}",
                    "batch_label": batch_label,
                    "params": {
                        "source": "asset_character_upload",
                        "image": {
                            "filepath": entry["filepath"],
                            "filename": entry["filename"],
                            "character": character,
                            "outfit": CHARACTER_UPLOAD_OUTFIT,
                            "expression": CHARACTER_UPLOAD_EXPRESSION,
                            "background_tags": background_tags,
                        },
                    },
                }
                for entry in saved
            ]
            created = await server.queue_manager.add_items_batch(items_spec)
            return web.json_response({
                "success": True,
                "character": character,
                "outfit": CHARACTER_UPLOAD_OUTFIT,
                "expression": CHARACTER_UPLOAD_EXPRESSION,
                "uploaded": [entry["filename"] for entry in saved],
                "count": len(saved),
                "tagging": {
                    "batch_id": created[0].batch_id if created else None,
                    "count": len(created),
                },
            }, headers={"Cache-Control": "no-store"})
        except (ValueError, TypeError, KeyError, OSError, json.JSONDecodeError) as exc:
            return web.json_response({"error": str(exc)}, status=400)

    server.app.router.add_get("/api/asset/config", config)
    server.app.router.add_post("/api/asset/config", config)
    server.app.router.add_post("/api/asset/enqueue", enqueue)
    server.app.router.add_post("/api/asset/preview", preview)
    server.app.router.add_post("/api/asset/character-upload", character_upload)
