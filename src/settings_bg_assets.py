"""Local background asset workspace using isolated storage and the shared generation queue."""
from __future__ import annotations

import asyncio
import json
import os
import re
from urllib.parse import quote, unquote

from aiohttp import web

from bg_asset_style_presets import (
    BG_STYLE_DEFAULT_HEIGHT,
    BG_STYLE_DEFAULT_WIDTH,
    BgAssetStylePresets,
    extract_bg_style_from_asset_style,
    register_bg_style_preset_runtime,
)
from bg_asset_presets import (
    BG_I2I_DEFAULT_DENOISE,
    _validate_folder_segment,
    build_bg_i2i_image_entry,
    register_bg_theme_preset_runtime,
)

READ_PATHS = {
    "tags", "themes", "style-presets", "theme-presets", "theme-maker", "postprocess",
    "model/presets", "model/workflow", "model/sampling-options",
}
WRITE_PATHS = {
    "tags", "style-presets", "style-presets/import-from-asset", "theme-presets", "theme-maker",
    "postprocess", "postprocess/preview",
    "theme/bulk-update",
    "lore-theme/generate", "lore-theme/cancel", "lore-theme/save",
    "model/presets", "model/workflow", "model/style-ref/upload", "model/style-ref/import",
    "upload_reference",
    "preview", "enqueue",
    "set_representative", "delete_image", "delete_combination", "delete_place",
    "name_mapping", "name_mapping/validate", "name_mapping/llm",
    "export", "export/validate",
}
MEDIA_PATTERN = r"/api/bg-asset/(?:theme-maker/image|i2i-reference/[^/]+|model/style-ref/image/[a-f0-9]{24}\.png|themes/[^/]+/places/[^/]+/atmospheres/[^/]+/images/[^/]+)"

_BG_TO_ASSET_BLOCK = {"theme": "character", "place": "outfit", "atmosphere": "expression"}
_ASSET_TO_BG_BLOCK = {value: key for key, value in _BG_TO_ASSET_BLOCK.items()}


def allowed_bg_asset_path(path: str, method: str) -> bool:
    if not path.startswith("/api/bg-asset/"):
        return False
    name = path[len("/api/bg-asset/"):]
    if method in {"GET", "HEAD"}:
        return (
            name in READ_PATHS
            or bool(re.fullmatch(MEDIA_PATTERN, path))
            or bool(re.fullmatch(
                r"(?:themes/[^/]+/(?:gallery|places/[^/]+/atmospheres/[^/]+/images)|name_mapping/[^/]+|base-link/[^/]+)",
                name,
            ))
        )
    return method == "POST" and name in WRITE_PATHS


def _to_asset_mapping(mapping: dict | None) -> dict:
    if not isinstance(mapping, dict):
        return {}
    result = dict(mapping)
    if "place_mapping" in result and "outfit_mapping" not in result:
        result["outfit_mapping"] = result.pop("place_mapping")
    if "atmosphere_mapping" in result and "expression_mapping" not in result:
        result["expression_mapping"] = result.pop("atmosphere_mapping")
    if "places" in result and "outfits" not in result:
        result["outfits"] = result.pop("places")
    if "atmospheres" in result and "expressions" not in result:
        result["expressions"] = result.pop("atmospheres")
    if isinstance(result.get("naming_order"), list):
        result["naming_order"] = [
            _BG_TO_ASSET_BLOCK.get(block, block) for block in result["naming_order"]
        ]
    if isinstance(result.get("naming_enabled"), dict):
        result["naming_enabled"] = {
            _BG_TO_ASSET_BLOCK.get(key, key): value
            for key, value in result["naming_enabled"].items()
        }
    return result


def _to_bg_mapping(info: dict) -> dict:
    result = dict(info)
    if "character" in result:
        result["theme"] = result.pop("character")
    if "outfits" in result:
        result["places"] = result.pop("outfits")
    if "expressions" in result:
        result["atmospheres"] = result.pop("expressions")
    if "outfit_expressions" in result:
        result["place_atmospheres"] = result.pop("outfit_expressions")
    if "outfit_mapping" in result:
        result["place_mapping"] = result.pop("outfit_mapping")
    if "expression_mapping" in result:
        result["atmosphere_mapping"] = result.pop("expression_mapping")
    if isinstance(result.get("naming_order"), list):
        result["naming_order"] = [
            _ASSET_TO_BG_BLOCK.get(block, block) for block in result["naming_order"]
        ]
    if isinstance(result.get("naming_enabled"), dict):
        result["naming_enabled"] = {
            _ASSET_TO_BG_BLOCK.get(key, key): value
            for key, value in result["naming_enabled"].items()
        }
    return result


def _localize_bg_text(text: str) -> str:
    if not isinstance(text, str):
        return text
    return (
        text.replace("캐릭터·복장·표정", "테마·장소·시간대")
        .replace("복장·표정", "장소·시간대")
        .replace("캐릭터", "테마")
        .replace("복장", "장소")
        .replace("표정", "시간대")
    )


def _to_bg_plan(plan: dict) -> dict:
    result = dict(plan)
    if "character" in result:
        result["theme"] = result.pop("character")
    if "character_count" in result:
        result["theme_count"] = result.pop("character_count")
    if "characters" in result and isinstance(result["characters"], list):
        result["themes"] = [_to_bg_plan(item) for item in result.pop("characters")]
    for issue_key in ("errors", "warnings"):
        if isinstance(result.get(issue_key), list):
            result[issue_key] = [
                {
                    **issue,
                    "message": _localize_bg_text(issue.get("message", "")),
                    **({"resolution": _localize_bg_text(issue["resolution"])} if issue.get("resolution") else {}),
                }
                for issue in result[issue_key]
            ]
    if isinstance(result.get("files"), list):
        result["files"] = [
            {
                **item,
                "place": item.get("outfit", ""),
                "atmosphere": item.get("expression", ""),
            }
            for item in result["files"]
        ]
    return result


def register_bg_assets(app: web.Application, runtime, schema: str) -> None:
    async def gateway(request: web.Request) -> web.StreamResponse:
        path = "/api/bg-asset/" + request.match_info["path"]
        media = request.method in {"GET", "HEAD"} and re.fullmatch(MEDIA_PATTERN, path)
        if not media:
            if request.headers.get("X-Comfy-Settings") != "1":
                raise web.HTTPForbidden()
            if request.headers.get("X-Comfy-Settings-Schema") != schema:
                return web.json_response({"error": "배경 에셋 화면을 새로고침해 주세요."}, status=409)
        return await runtime.request(request, bg_asset=True)

    app.router.add_route("GET", "/api/bg-asset/{path:.+}", gateway)
    app.router.add_route("POST", "/api/bg-asset/{path:.+}", gateway)


def register_bg_asset_runtime(server) -> None:
    from asset_model_presets import BgAssetModelPresets, apply_asset_preset
    from asset_postprocess import (
        BG_ASSET_WORKSPACE,
        has_active_postprocess,
        register_postprocess_runtime,
        render_cached_postprocess_file,
    )
    from asset_style_presets import AssetStylePresets
    from illustration_model_presets import register_model_preset_runtime
    from settings_asset_export import build_asset_export_plans, export_asset_zip, public_asset_export_plans

    presets = BgAssetModelPresets(server.CONFIG_FILE)
    asset_styles = AssetStylePresets(server.CONFIG_FILE)
    register_model_preset_runtime(server, "bg-asset/model", presets)
    bg_styles = register_bg_style_preset_runtime(server)
    register_bg_theme_preset_runtime(server)
    bg_mode = server.bg_asset_mode
    bg_postprocess_store = register_postprocess_runtime(
        server, BG_ASSET_WORKSPACE, bg_mode, prefix="/api/bg-asset/postprocess"
    )

    async def sampling_options(request: web.Request) -> web.Response:
        from settings_illustration import illustration_sampling_options
        return web.json_response(
            await illustration_sampling_options(server, "asset_generation"),
            headers={"Cache-Control": "no-store"},
        )

    server.app.router.add_get("/api/bg-asset/model/sampling-options", sampling_options)

    def prepare_bg_job(job: dict, style_presets: list[dict]) -> dict:
        from asset_json import asset_document, build_asset_moss_document, prepare_bg_i2i

        theme = job.get("theme") or job.get("character") or ""
        if not isinstance(theme, str) or not theme.strip():
            raise ValueError("테마를 선택하세요.")
        theme = theme.strip()
        job["theme"] = theme
        job["character"] = theme

        place = job.get("place") if "place" in job else job.get("outfit")
        if place is not None and not isinstance(place, str):
            raise ValueError("장소 이름을 확인하세요.")
        clean_place = place.strip() if isinstance(place, str) and place.strip() else "기본"
        job["place"] = clean_place
        job["outfit"] = clean_place

        time_val = job.get("time") if "time" in job else (job.get("atmosphere") if "atmosphere" in job else job.get("expression"))
        if time_val is not None and not isinstance(time_val, str):
            raise ValueError("시간대 이름을 확인하세요.")
        clean_time = time_val.strip() if isinstance(time_val, str) and time_val.strip() else "기본"
        job["time"] = clean_time
        job["atmosphere"] = clean_time
        job["expression"] = clean_time
        job["appearance"] = "기본"

        identity = job.get("asset_style_preset_id")
        if not isinstance(identity, str) or not any(preset["id"] == identity for preset in style_presets):
            raise ValueError("사용할 배경 그림체 프리셋을 선택하세요. 삭제된 프리셋은 다시 선택해야 합니다.")

        snapshot = presets.snapshot(server.app_config, job.get("asset_model_preset_id"))
        profile = snapshot["config"]["asset_workflow_type"]
        if profile != job.get("asset_workflow_type"):
            raise ValueError("그림 모델이 변경되었습니다. 프리셋을 다시 선택하세요.")
        if not snapshot.get("workflows", {}).get(profile, {}).get("fields"):
            raise ValueError("그림 모델 설정에서 워크플로우를 선택하세요.")

        document = job.get("prompt_json")
        if not isinstance(document, dict) or not isinstance(document.get("fields"), dict) or not document["fields"]:
            raise ValueError("프롬프트 JSON의 fields를 확인하세요.")
        document = asset_document(json.dumps(document, ensure_ascii=False, allow_nan=False))
        fields = document["fields"]
        # Enforce background-only invariants: full canvas, no character masks/detailers/references.
        fields["MULTI_CHAR"] = {"enable": False}
        fields["FACE_LORA_ACTIVATE"] = False
        fields["FACE_LORA_DATA"] = {"list": []}
        fields["ANIMA_FD_ACTIVATE"] = False
        fields["ANIMA_HD_ACTIVATE"] = False
        fields["ANIMA_ED_ACTIVATE"] = False
        fields["FD_ACTIVATE"] = False
        fields["HD_ACTIVATE"] = False
        fields["ED_ACTIVATE"] = False
        if "SNAPREF" in fields and isinstance(fields["SNAPREF"], dict):
            fields["SNAPREF"] = {**fields["SNAPREF"], "enabled": False, "references": []}
            fields["SNAPREF"].pop("dir", None)
        else:
            fields["SNAPREF"] = {"enabled": False, "references": []}

        prepare_bg_i2i(document, job)

        if profile == "moss":
            from modes.moss_prompt_builder import MossPromptBuilder

            document = build_asset_moss_document(document, snapshot)
            job["snapref_enabled"] = False
            job["snapref_embedded"] = True
            job["prompt_json"] = document
            job["positive_prompt"] = MossPromptBuilder.dumps(document)
            job["negative_prompt"] = ""
            job.update(
                asset_workspace="bg_asset",
                face_id_enabled=False,
                style_ref_enabled=False,
                pose_enabled=False,
                hrf_control_net=False,
            )
            return snapshot

        document = apply_asset_preset(document, snapshot)
        # Keep model-level style_ref working via embed_style_ref_images, while character snapref stays off.
        from style_ref_storage import embed_style_ref_images

        embed_style_ref_images(document)
        job["snapref_enabled"] = False
        job["snapref_embedded"] = True
        job["prompt_json"] = document
        job["positive_prompt"] = json.dumps(document, ensure_ascii=False, allow_nan=False)
        job["negative_prompt"] = ""
        job.update(
            asset_workspace="bg_asset",
            face_id_enabled=False,
            style_ref_enabled=False,
            pose_enabled=False,
            hrf_control_net=False,
        )
        return snapshot

    async def import_style_from_asset(request: web.Request) -> web.Response:
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise ValueError("가져오기 요청을 확인하세요.")
            source_id = body.get("asset_style_preset_id")
            if not isinstance(source_id, str) or not source_id.strip():
                raise ValueError("가져올 에셋 그림체 프리셋을 선택하세요.")
            source = next(
                (item for item in asset_styles.read()["presets"] if item["id"] == source_id),
                None,
            )
            if source is None:
                raise ValueError("선택한 에셋 그림체 프리셋을 찾을 수 없습니다.")
            keep_resolution = bool(body.get("keep_resolution", False))
            converted = extract_bg_style_from_asset_style(
                source["settings"],
                keep_resolution=keep_resolution,
                default_width=BG_STYLE_DEFAULT_WIDTH,
                default_height=BG_STYLE_DEFAULT_HEIGHT,
            )
            save_as = body.get("save_as")
            if isinstance(save_as, str) and save_as.strip():
                catalog = bg_styles.catalog()
                saved = bg_styles.change({
                    "action": "create",
                    "name": save_as.strip(),
                    "settings": converted,
                    "revision": catalog["revision"],
                })
                return web.json_response(
                    {**saved, "imported_settings": converted, "source_name": source["name"]},
                    headers={"Cache-Control": "no-store"},
                )
            return web.json_response(
                {"success": True, "imported_settings": converted, "source_name": source["name"]},
                headers={"Cache-Control": "no-store"},
            )
        except (ValueError, TypeError, OSError) as exc:
            return web.json_response({"error": str(exc)}, status=400)

    async def tags_handler(request: web.Request) -> web.Response:
        try:
            if request.method == "GET":
                return web.json_response(bg_mode.get_tags(), headers={"Cache-Control": "no-store"})
            body = await request.json()
            if not isinstance(body, dict):
                raise ValueError("배경 태그 요청을 확인하세요.")
            action = body.get("action")
            if action == "add_theme":
                result = bg_mode.add_character(body.get("name", ""))
            elif action == "duplicate_theme":
                result = bg_mode.duplicate_character(body.get("source", ""), body.get("name", ""))
            elif action == "remove_theme":
                result = bg_mode.remove_character(
                    body.get("name", ""),
                    builtin_unlock_token=body.get("builtin_unlock_token"),
                )
            elif action == "update_theme":
                result = bg_mode.update_theme(
                    body.get("name", ""),
                    fields=body.get("fields"),
                    group=body.get("group"),
                    builtin_unlock_token=body.get("builtin_unlock_token"),
                )
            elif action == "set_place_group":
                result = bg_mode.set_outfit_group(
                    body.get("theme", ""),
                    body.get("src_place", ""),
                    body.get("src_atmosphere", ""),
                    body.get("tgt_place", ""),
                    body.get("tgt_atmosphere", ""),
                )
            elif action == "ungroup_place":
                result = bg_mode.ungroup_outfit(
                    body.get("theme", ""),
                    body.get("place", ""),
                    body.get("atmosphere", ""),
                )
            else:
                raise ValueError("지원하지 않는 배경 태그 요청입니다.")
            status = 409 if result.get("conflict") else (200 if result.get("success", True) else 400)
            return web.json_response(result, status=status, headers={"Cache-Control": "no-store"})
        except (ValueError, TypeError, OSError) as exc:
            return web.json_response({"success": False, "error": str(exc)}, status=400)

    async def themes_list(request: web.Request) -> web.Response:
        raw_reps = bg_mode.get_characters_representative()
        representatives = {
            theme: {
                "place": rep.get("outfit", ""),
                "atmosphere": rep.get("expression", ""),
                "filename": rep.get("filename", ""),
            }
            for theme, rep in raw_reps.items()
        }
        return web.json_response(
            {"themes": bg_mode.list_themes(), "representatives": representatives},
            headers={"Cache-Control": "no-store"},
        )

    async def theme_gallery(request: web.Request) -> web.Response:
        theme = unquote(request.match_info.get("theme", ""))
        raw_gallery = bg_mode.list_character_gallery(theme)
        gallery = [
            {
                **item,
                "place": item.get("outfit", ""),
                "atmosphere": item.get("expression", ""),
            }
            for item in raw_gallery
        ]
        raw_groups = bg_mode.get_outfit_groups(theme)
        place_groups = {
            gid: [{"place": m.get("outfit", ""), "atmosphere": m.get("expression", "")} for m in members]
            for gid, members in raw_groups.items()
        }
        return web.json_response(
            {"theme": theme, "gallery": gallery, "place_groups": place_groups},
            headers={"Cache-Control": "no-store"},
        )

    async def theme_images(request: web.Request) -> web.Response:
        theme = unquote(request.match_info.get("theme", ""))
        place = unquote(request.match_info.get("place", ""))
        atmosphere = unquote(request.match_info.get("atmosphere", ""))
        result = bg_mode.list_images(theme, place, atmosphere)
        return web.json_response(result, headers={"Cache-Control": "no-store"})

    async def theme_image_file(request: web.Request) -> web.StreamResponse:
        theme = unquote(request.match_info.get("theme", ""))
        place = unquote(request.match_info.get("place", ""))
        atmosphere = unquote(request.match_info.get("atmosphere", ""))
        filename = unquote(request.match_info.get("filename", ""))
        if not filename or os.path.basename(filename) != filename or filename in (".", ".."):
            raise web.HTTPBadRequest()
        path = bg_mode.get_image_path(theme, place, atmosphere, filename)
        if not path:
            raise web.HTTPNotFound()
        if request.query.get("postprocess") == "1":
            postprocess_settings = await asyncio.to_thread(bg_postprocess_store.read)
            if has_active_postprocess(BG_ASSET_WORKSPACE, postprocess_settings):
                image_data, content_type, applied = await asyncio.to_thread(
                    render_cached_postprocess_file,
                    path,
                    BG_ASSET_WORKSPACE,
                    postprocess_settings,
                )
                return web.Response(
                    body=image_data,
                    content_type=content_type,
                    headers={
                        "Cache-Control": "no-store",
                        "X-Postprocess-Applied": "1" if applied else "0",
                    },
                )
        return web.FileResponse(path, headers={"Cache-Control": "no-store"})

    async def set_representative(request: web.Request) -> web.Response:
        try:
            body = await request.json()
            result = bg_mode.set_representative(
                body.get("theme", ""),
                body.get("place", ""),
                body.get("atmosphere", ""),
                body.get("filename", ""),
            )
            return web.json_response(result, headers={"Cache-Control": "no-store"})
        except (ValueError, TypeError, OSError) as exc:
            return web.json_response({"success": False, "error": str(exc)}, status=400)

    async def delete_image(request: web.Request) -> web.Response:
        try:
            body = await request.json()
            result = bg_mode.delete_image(
                body.get("theme", ""),
                body.get("place", ""),
                body.get("atmosphere", ""),
                body.get("filename", ""),
            )
            return web.json_response(result, status=200 if result.get("success") else 400, headers={"Cache-Control": "no-store"})
        except (ValueError, TypeError, OSError) as exc:
            return web.json_response({"success": False, "error": str(exc)}, status=400)

    async def delete_combination(request: web.Request) -> web.Response:
        try:
            body = await request.json()
            result = bg_mode.delete_combination(
                body.get("theme", ""),
                body.get("place", ""),
                body.get("atmosphere", ""),
            )
            return web.json_response(result, status=200 if result.get("success") else 400, headers={"Cache-Control": "no-store"})
        except (ValueError, TypeError, OSError) as exc:
            return web.json_response({"success": False, "error": str(exc)}, status=400)

    async def delete_place(request: web.Request) -> web.Response:
        try:
            body = await request.json()
            result = bg_mode.delete_place(
                body.get("theme", ""),
                body.get("place", ""),
            )
            return web.json_response(result, status=200 if result.get("success") else 400, headers={"Cache-Control": "no-store"})
        except (ValueError, TypeError, OSError) as exc:
            return web.json_response({"success": False, "error": str(exc)}, status=400)

    async def preview(request: web.Request) -> web.Response:
        try:
            job = await request.json()
            if not isinstance(job, dict):
                raise ValueError("프롬프트 JSON을 확인하세요.")
            prepare_bg_job(job, bg_styles.read()["presets"])
            return web.json_response(job, headers={"Cache-Control": "no-store"})
        except (ValueError, TypeError, KeyError, OSError) as exc:
            return web.json_response({"error": str(exc)}, status=400)

    base_link_batches: dict[str, dict] = {}
    _max_base_link_batches = 50

    def _public_base_link_state(state: dict) -> dict:
        linked = list(state["linked"])
        failed = list(state["failed"])
        pending = list(state["pending"])
        total = int(state["total"])
        return {
            "success": True,
            "batch_id": state["batch_id"],
            "theme": state["theme"],
            "base_time": state["base_time"],
            "total": total,
            "linked_count": len(linked),
            "failed_count": len(failed),
            "pending_count": len(pending),
            "completed": len(linked) + len(failed),
            "done": len(pending) == 0,
            "linked": linked,
            "failed": failed,
            "pending": pending,
        }

    def _mark_base_link_linked(batch_id: str, place_name: str, image_entry: dict) -> None:
        state = base_link_batches.get(batch_id)
        if not state:
            return
        if place_name in state["pending"]:
            state["pending"].remove(place_name)
        state["linked"] = [entry for entry in state["linked"] if entry.get("place") != place_name]
        state["linked"].append({
            "place": place_name,
            "filename": image_entry["name"],
            "preview_url": image_entry["preview_url"],
        })

    def _mark_base_link_failed(batch_id: str, place_name: str, message: str) -> None:
        state = base_link_batches.get(batch_id)
        if not state:
            return
        if place_name in state["pending"]:
            state["pending"].remove(place_name)
        state["failed"] = [entry for entry in state["failed"] if entry.get("place") != place_name]
        state["failed"].append({
            "place": place_name,
            "error": message,
        })

    async def base_link_status(request: web.Request) -> web.Response:
        batch_id = unquote(request.match_info.get("batch_id", "")).strip()
        state = base_link_batches.get(batch_id)
        if state is None:
            return web.json_response({"success": False, "error": "베이스 I2I 연결 작업 정보를 찾을 수 없습니다."}, status=404)
        return web.json_response(_public_base_link_state(state), headers={"Cache-Control": "no-store"})

    async def enqueue(request: web.Request) -> web.Response:
        from pathlib import Path
        from bg_asset_presets import _normalize_i2i_denoise

        body = await request.json()
        jobs = body.get("jobs") if isinstance(body, dict) else None
        if not isinstance(jobs, list) or not jobs:
            return web.json_response({"error": "생성할 배경 에셋이 없습니다."}, status=400)
        auto_link_i2i = bool(body.get("auto_link_i2i", False)) if isinstance(body, dict) else False
        raw_link_denoise = body.get("auto_link_i2i_denoise") if isinstance(body, dict) else None
        link_denoise = None
        if auto_link_i2i and raw_link_denoise is not None:
            try:
                link_denoise = _normalize_i2i_denoise(raw_link_denoise, strict=True)
            except (ValueError, TypeError) as exc:
                return web.json_response({"error": str(exc)}, status=400)
        try:
            style_presets = bg_styles.read()["presets"]
        except (ValueError, TypeError, OSError) as exc:
            return web.json_response({"error": str(exc)}, status=400)
        snapshots = []
        for index, job in enumerate(jobs):
            if not isinstance(job, dict):
                return web.json_response({"error": f"항목 {index + 1}: 생성할 배경 에셋을 확인하세요."}, status=400)
            if job.get("asset_workflow_type") not in {"anima_only", "moss"}:
                return web.json_response({"error": f"항목 {index + 1}: ANIMA 또는 MOSS 배경 워크플로우를 선택하세요."}, status=400)
            try:
                snapshot = prepare_bg_job(job, style_presets)
                snapshots.append(snapshot)
            except (ValueError, TypeError, KeyError, OSError) as exc:
                return web.json_response({"error": f"항목 {index + 1}: {exc}"}, status=400)

        if auto_link_i2i:
            catalog = bg_mode.theme_presets.catalog()
            for index, job in enumerate(jobs):
                theme_name = job["theme"]
                if theme_name not in catalog["themes"]:
                    return web.json_response({"error": f"항목 {index + 1}: 테마를 찾을 수 없습니다 ({theme_name})."}, status=400)
                if catalog["scopes"].get(theme_name) == "builtin":
                    return web.json_response(
                        {"error": "배포 테마에는 로컬 I2I 기준 이미지를 저장할 수 없습니다. 새 테마 만들기 또는 사용자 테마를 사용하세요."},
                        status=400,
                    )
                if job["place"] not in (catalog["themes"][theme_name].get("places") or {}):
                    return web.json_response({"error": f"항목 {index + 1}: 장소를 찾을 수 없습니다 ({job['place']})."}, status=400)

        label = body.get("label") or "배경 생성"
        items = await server.queue_manager.add_items_batch([
            {
                "type": "asset_generation",
                "label": f"배경 · {job['theme']}/{job['place']}/{job['time']}",
                "batch_label": label,
                "params": {"body": job, "asset_model_preset": snapshot, "asset_workspace": "bg_asset"},
            }
            for job, snapshot in zip(jobs, snapshots)
        ])
        batch_id = items[0].batch_id

        if auto_link_i2i:
            while len(base_link_batches) >= _max_base_link_batches:
                base_link_batches.pop(next(iter(base_link_batches)))
            base_link_batches[batch_id] = {
                "batch_id": batch_id,
                "theme": jobs[0]["theme"],
                "base_time": jobs[0]["time"],
                "total": len(items),
                "linked": [],
                "failed": [],
                "pending": [job["place"] for job in jobs],
            }

            def _attach_completion_hook(queue_item, job_snapshot):
                theme_name = job_snapshot["theme"]
                place_name = job_snapshot["place"]
                time_name = job_snapshot["time"]

                def _on_done(fut):
                    try:
                        result = fut.result()
                    except asyncio.CancelledError:
                        _mark_base_link_failed(batch_id, place_name, "베이스 이미지 생성이 취소되었습니다.")
                        return
                    except BaseException as exc:
                        _mark_base_link_failed(batch_id, place_name, str(exc) or "베이스 이미지 생성에 실패했습니다.")
                        return

                    if not isinstance(result, dict) or not result.get("success"):
                        err = (result.get("error") if isinstance(result, dict) else "") or "베이스 이미지 생성에 실패했습니다."
                        _mark_base_link_failed(batch_id, place_name, str(err))
                        return

                    filename = str(result.get("filename") or "").strip()
                    if not filename:
                        _mark_base_link_failed(batch_id, place_name, "생성된 베이스 이미지 파일명을 확인할 수 없습니다.")
                        return
                    local_path = bg_mode.get_image_path(theme_name, place_name, time_name, filename)
                    if not local_path or not os.path.isfile(local_path):
                        _mark_base_link_failed(batch_id, place_name, "생성된 베이스 이미지 파일을 찾을 수 없습니다.")
                        return

                    image_entry = build_bg_i2i_image_entry(
                        theme_name,
                        place_name,
                        time_name,
                        filename,
                        local_path,
                    )
                    try:
                        bg_mode.theme_presets.attach_place_i2i_image(
                            theme_name,
                            place_name,
                            image_entry,
                            enable_i2i=True,
                            denoise=link_denoise,
                        )
                        bg_mode.load_tags()
                    except Exception as exc:
                        _mark_base_link_failed(batch_id, place_name, f"I2I 기준 이미지 연결 실패: {exc}")
                        return
                    _mark_base_link_linked(batch_id, place_name, image_entry)

                queue_item.completion_future.add_done_callback(_on_done)

            for queue_item, job in zip(items, jobs):
                _attach_completion_hook(queue_item, job)

        return web.json_response({
            "success": True,
            "count": len(items),
            "batch_id": batch_id,
            "auto_link_i2i": auto_link_i2i,
        })

    # ─── 배경 이름 치환 & 익스포트 ───
    async def name_mapping_get(request: web.Request) -> web.Response:
        theme = unquote(request.match_info.get("theme", ""))
        if not theme:
            return web.json_response({"error": "테마 이름 필요"}, status=400)
        try:
            info = bg_mode.get_character_export_info(theme)
            return web.json_response(_to_bg_mapping(info), headers={"Cache-Control": "no-store"})
        except Exception as exc:
            return web.json_response({"error": f"이름 치환 정보 조회 실패: {exc}"}, status=500)

    async def name_mapping_post(request: web.Request) -> web.Response:
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise ValueError("요청 본문은 JSON 객체여야 합니다.")
            theme = body.get("theme", "")
            if not isinstance(theme, str) or not theme.strip():
                raise ValueError("테마 이름 필요")
            converted = _to_asset_mapping(body)
            result = bg_mode.save_character_name_mapping(
                theme,
                converted.get("export_name", ""),
                converted.get("outfit_mapping", {}),
                converted.get("expression_mapping", {}),
                converted.get("export_format", "webp"),
                converted.get("export_quality", 90),
                converted.get("naming_order"),
                converted.get("naming_enabled"),
            )
            return web.json_response(result, headers={"Cache-Control": "no-store"})
        except ValueError as exc:
            return web.json_response({"success": False, "error": _localize_bg_text(str(exc))}, status=400)

    async def name_mapping_validate(request: web.Request) -> web.Response:
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise ValueError("요청 본문은 JSON 객체여야 합니다.")
            theme = body.get("theme", "")
            if not isinstance(theme, str) or not theme.strip():
                raise ValueError("테마 이름이 필요합니다.")
            places = body.get("places")
            atmospheres = body.get("atmospheres")
            mapping = _to_asset_mapping(body.get("mapping", {}))
            plan = bg_mode.build_character_export_plan(theme, places, atmospheres, mapping_override=mapping)
            public_plan = _to_bg_plan(bg_mode.public_export_plan(plan))
            return web.json_response(public_plan, status=200 if public_plan["success"] else 409, headers={"Cache-Control": "no-store"})
        except ValueError as exc:
            return web.json_response(
                {"success": False, "error": _localize_bg_text(str(exc)),
                 "errors": [{"code": "invalid_request", "message": _localize_bg_text(str(exc))}]},
                status=400,
            )

    async def name_mapping_llm(request: web.Request) -> web.Response:
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise ValueError("요청 본문은 JSON 객체여야 합니다.")
            adapted_body = {
                "mode": body.get("mode", ""),
                "character": body.get("theme", ""),
                "outfits": body.get("places"),
                "expressions": body.get("atmospheres"),
                "mapping": _to_asset_mapping(body.get("mapping", {})),
            }
            response = await server.run_asset_name_mapping_llm(adapted_body, target_mode=bg_mode)
            payload = json.loads(response.text)
            if isinstance(payload.get("mapping"), dict):
                payload["mapping"] = _to_bg_mapping(payload["mapping"])
            if isinstance(payload.get("validation"), dict):
                payload["validation"] = _to_bg_plan(payload["validation"])
            if isinstance(payload.get("error"), str):
                payload["error"] = _localize_bg_text(payload["error"])
            if isinstance(payload.get("message"), str):
                payload["message"] = _localize_bg_text(payload["message"])
            return web.json_response(payload, status=response.status, headers={"Cache-Control": "no-store"})
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            return web.json_response({"success": False, "error": _localize_bg_text(str(exc))}, status=400)

    async def export_handler(request: web.Request) -> web.Response:
        try:
            body = await request.json()
            raw_themes = body.get("themes") if isinstance(body, dict) else None
            if not isinstance(raw_themes, list) or not raw_themes:
                raise ValueError("내보낼 테마를 하나 이상 선택하세요.")
            adapted = {
                "characters": [
                    {
                        "character": entry.get("theme", ""),
                        "outfits": entry.get("places"),
                        "expressions": entry.get("atmospheres"),
                        "mapping": _to_asset_mapping(entry.get("mapping")),
                    }
                    for entry in raw_themes
                    if isinstance(entry, dict)
                ]
            }
            plans = await asyncio.to_thread(build_asset_export_plans, bg_mode, adapted)
            public = _to_bg_plan(public_asset_export_plans(bg_mode, plans))
            if not public["success"] or request.path.endswith("/validate"):
                return web.json_response(public, status=200 if public["success"] else 409, headers={"Cache-Control": "no-store"})
            postprocess_settings = None
            if isinstance(body, dict) and bool(body.get("apply_postprocess", False)):
                postprocess_settings = await asyncio.to_thread(bg_postprocess_store.read)
            content = await asyncio.to_thread(
                export_asset_zip,
                bg_mode,
                plans,
                postprocess_settings=postprocess_settings,
            )
            filename = f"{plans[0][0]}.zip" if len(plans) == 1 else "배경 에셋 익스포트.zip"
            return web.Response(
                body=content,
                content_type="application/zip",
                headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(filename)}"},
            )
        except (ValueError, TypeError, OSError, RuntimeError) as exc:
            return web.json_response({"success": False, "error": _localize_bg_text(str(exc))}, status=400)

    async def upload_reference(request: web.Request) -> web.Response:
        import hashlib
        from io import BytesIO
        from pathlib import Path
        import time as _time
        import uuid as _uuid
        from PIL import Image, ImageOps
        from configuration import settings_lock
        from json_store import write_json_atomic

        max_bytes = 12 * 1024 * 1024
        try:
            reader = await request.multipart()
            images_data: list[tuple[str, bytes]] = []
            async for part in reader:
                if part.name != "image":
                    continue
                chunks = []
                size = 0
                while True:
                    chunk = await part.read_chunk(64 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > max_bytes:
                        raise ValueError("이미지 한 장은 12MB를 넘을 수 없습니다.")
                    chunks.append(chunk)
                raw = b"".join(chunks)
                if raw:
                    images_data.append((part.filename or "i2i_reference.png", raw))
            if not images_data:
                raise ValueError("업로드할 이미지를 선택하세요.")

            save_dir = Path(bg_mode.asset_data_dir) / "i2i_references"
            save_dir.mkdir(parents=True, exist_ok=True)
            hash_file = save_dir / "_upload_hashes.json"

            results = []
            with settings_lock(hash_file):
                hash_map = {}
                if hash_file.is_file():
                    try:
                        loaded = json.loads(hash_file.read_text(encoding="utf-8"))
                        if isinstance(loaded, dict):
                            hash_map = loaded
                    except Exception:
                        hash_map = {}

                changed = False
                for orig_filename, raw in images_data[:1]:
                    try:
                        with Image.open(BytesIO(raw)) as opened:
                            img = ImageOps.exif_transpose(opened).copy()
                    except Exception as exc:
                        raise ValueError("지원하지 않는 이미지 파일입니다. PNG, JPEG, WebP 파일을 선택하세요.") from exc

                    content_hash = hashlib.sha256(img.convert("RGB").tobytes()).hexdigest()
                    cached_name = hash_map.get(content_hash)
                    if isinstance(cached_name, str) and cached_name and (save_dir / cached_name).is_file():
                        stored_name = cached_name
                    else:
                        stored_name = f"{int(_time.time())}_{_uuid.uuid4().hex[:6]}.webp"
                        target_path = save_dir / stored_name
                        try:
                            save_img = img if img.mode == "RGBA" else img.convert("RGB")
                            save_img.save(target_path, format="WEBP", quality=92, method=4)
                        except Exception:
                            stored_name = f"{int(_time.time())}_{_uuid.uuid4().hex[:6]}.png"
                            target_path = save_dir / stored_name
                            img.convert("RGB").save(target_path, format="PNG")
                        hash_map[content_hash] = stored_name
                        changed = True

                    local_path = str((save_dir / stored_name).resolve())
                    results.append({
                        "name": stored_name,
                        "local_path": local_path,
                        "orig_filename": os.path.basename(orig_filename) or stored_name,
                        "preview_url": f"/api/bg-asset/i2i-reference/{quote(stored_name)}",
                    })

                if changed:
                    write_json_atomic(hash_file, hash_map)

            return web.json_response({"success": True, "images": results}, headers={"Cache-Control": "no-store"})
        except (ValueError, TypeError, OSError) as exc:
            return web.json_response({"success": False, "error": str(exc)}, status=400)

    async def i2i_reference_file(request: web.Request) -> web.StreamResponse:
        from pathlib import Path

        filename = unquote(request.match_info["filename"])
        if not filename or filename.startswith("_") or "/" in filename or "\\" in filename or ".." in filename:
            raise web.HTTPNotFound()
        root = (Path(bg_mode.asset_data_dir) / "i2i_references").resolve()
        candidate = (root / filename).resolve()
        if candidate.parent != root or not candidate.is_file():
            raise web.HTTPNotFound()
        return web.FileResponse(candidate, headers={"Cache-Control": "no-store"})

    async def bulk_update_theme(request: web.Request) -> web.Response:
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise ValueError("일괄 설정 요청을 확인하세요.")
            theme_name = _validate_folder_segment(body.get("theme"), "테마 이름")

            raw_i2i = body.get("i2i")
            if raw_i2i is None:
                raw_i2i = {}
            if not isinstance(raw_i2i, dict):
                raise ValueError("I2I 일괄 연결 설정 형식을 확인하세요.")
            i2i_mode = str(raw_i2i.get("mode") or "none").strip()
            if i2i_mode not in {"none", "missing_only", "overwrite", "denoise_only"}:
                raise ValueError("I2I 일괄 연결 방식을 확인하세요.")
            i2i_denoise = raw_i2i.get("denoise", BG_I2I_DEFAULT_DENOISE)

            clean_source_time = ""
            resolved_entries: dict[str, dict | None] | None = None
            raw_places = body.get("places")

            if i2i_mode in {"missing_only", "overwrite"}:
                raw_source_time = raw_i2i.get("source_time", "")
                if not isinstance(raw_source_time, str) or not raw_source_time.strip():
                    raise ValueError("기준 시간대를 선택하세요.")
                clean_source_time = _validate_folder_segment(raw_source_time, "기준 시간대")

                catalog = bg_mode.theme_presets.catalog()
                if theme_name not in catalog["themes"]:
                    raise ValueError("테마를 찾을 수 없습니다. 목록을 새로고침하세요.")
                if catalog["scopes"].get(theme_name) == "builtin":
                    raise ValueError(
                        "배포 테마에는 로컬 I2I 기준 이미지를 저장할 수 없습니다. 사용자 테마로 복제하거나 그룹을 변경해 주세요."
                    )

                places_map = catalog["themes"][theme_name].get("places") or {}
                if raw_places is None:
                    target_places = list(places_map.keys())
                else:
                    if not isinstance(raw_places, list) or not raw_places:
                        raise ValueError("적용할 장소를 하나 이상 선택하세요.")
                    target_places = list(dict.fromkeys(
                        item.strip() for item in raw_places if isinstance(item, str) and item.strip()
                    ))
                resolved_entries = {
                    place_name: bg_mode.resolve_place_i2i_image(theme_name, place_name, clean_source_time)
                    for place_name in target_places
                    if place_name in places_map
                }

            result = bg_mode.theme_presets.bulk_update_theme(
                theme_name,
                places=raw_places,
                add_times=body.get("add_times"),
                remove_times=body.get("remove_times"),
                i2i_mode=i2i_mode,
                i2i_denoise=i2i_denoise,
                i2i_place_images=resolved_entries,
                builtin_unlock_token=body.get("builtin_unlock_token"),
            )
            result["i2i"]["source_time"] = clean_source_time
            bg_mode.load_tags()
            return web.json_response(result, headers={"Cache-Control": "no-store"})
        except (ValueError, TypeError, OSError) as exc:
            return web.json_response({"success": False, "error": str(exc)}, status=400)

    server.app.router.add_post("/api/bg-asset/style-presets/import-from-asset", import_style_from_asset)
    server.app.router.add_post("/api/bg-asset/upload_reference", upload_reference)
    server.app.router.add_get("/api/bg-asset/i2i-reference/{filename}", i2i_reference_file)
    server.app.router.add_get("/api/bg-asset/tags", tags_handler)
    server.app.router.add_post("/api/bg-asset/tags", tags_handler)
    server.app.router.add_post("/api/bg-asset/theme/bulk-update", bulk_update_theme)
    server.app.router.add_get("/api/bg-asset/themes", themes_list)
    server.app.router.add_get("/api/bg-asset/themes/{theme}/gallery", theme_gallery)
    server.app.router.add_get("/api/bg-asset/themes/{theme}/places/{place}/atmospheres/{atmosphere}/images", theme_images)
    server.app.router.add_get("/api/bg-asset/themes/{theme}/places/{place}/atmospheres/{atmosphere}/images/{filename}", theme_image_file)
    server.app.router.add_post("/api/bg-asset/set_representative", set_representative)
    server.app.router.add_post("/api/bg-asset/delete_image", delete_image)
    server.app.router.add_post("/api/bg-asset/delete_combination", delete_combination)
    server.app.router.add_post("/api/bg-asset/delete_place", delete_place)
    server.app.router.add_post("/api/bg-asset/preview", preview)
    server.app.router.add_post("/api/bg-asset/enqueue", enqueue)
    server.app.router.add_get("/api/bg-asset/base-link/{batch_id}", base_link_status)
    server.app.router.add_get("/api/bg-asset/name_mapping/{theme}", name_mapping_get)
    server.app.router.add_post("/api/bg-asset/name_mapping", name_mapping_post)
    server.app.router.add_post("/api/bg-asset/name_mapping/validate", name_mapping_validate)
    server.app.router.add_post("/api/bg-asset/name_mapping/llm", name_mapping_llm)
    server.app.router.add_post("/api/bg-asset/export/validate", export_handler)
    server.app.router.add_post("/api/bg-asset/export", export_handler)

    from bg_asset_lore_theme import register_bg_lore_theme_runtime
    register_bg_lore_theme_runtime(server)

    if hasattr(server, "character_maker"):
        from bg_asset_maker import register_bg_theme_maker_runtime
        register_bg_theme_maker_runtime(server, lambda job: prepare_bg_job(job, bg_styles.read()["presets"]))
