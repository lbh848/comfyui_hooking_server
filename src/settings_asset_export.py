"""Export selected workshop characters through the existing asset export engine."""
from __future__ import annotations

import asyncio
import io
import shutil
import zipfile
from urllib.parse import quote

from aiohttp import web


def build_asset_export_plans(mode, body: dict) -> list[tuple[str, dict]]:
    entries = body.get("characters") if isinstance(body, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ValueError("내보낼 캐릭터를 하나 이상 선택하세요.")
    available = set(mode.list_characters())
    seen = set()
    filename_owners = {}
    plans = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("캐릭터 내보내기 설정을 확인하세요.")
        character = entry.get("character")
        if not isinstance(character, str) or character not in available:
            raise ValueError(f"내보낼 캐릭터를 찾을 수 없습니다: {character}")
        if character in seen:
            raise ValueError(f"같은 캐릭터가 중복 선택되었습니다: {character}")
        seen.add(character)
        mapping = entry.get("mapping")
        if mapping is not None and not isinstance(mapping, dict):
            raise ValueError(f"{character}: 이름 치환 규칙을 확인하세요.")
        plan = mode.build_character_export_plan(
            character, entry.get("outfits"), entry.get("expressions"), mapping_override=mapping,
        )
        if len(entries) > 1 and plan["success"]:
            for item in plan["files"]:
                filename = item["filename"]
                key = mode._normalize_export_collision_key(filename)
                owner = filename_owners.setdefault(key, character)
                if owner != character:
                    raise ValueError(
                        f"‘{owner}’와 ‘{character}’의 최종 파일명 ‘{filename}’이 겹칩니다. "
                        "이름 설정에서 두 캐릭터의 파일명을 다르게 지정하세요."
                    )
        plans.append((character, plan))
    return plans


def public_asset_export_plans(mode, plans: list[tuple[str, dict]]) -> dict:
    return {
        "success": all(plan["success"] for _, plan in plans),
        "character_count": len(plans),
        "file_count": sum(plan["file_count"] for _, plan in plans),
        "characters": [{"character": character, **mode.public_export_plan(plan)} for character, plan in plans],
    }


def export_asset_zip(mode, plans: list[tuple[str, dict]]) -> bytes:
    if len(plans) == 1:
        character, plan = plans[0]
        result = mode.export_character_zip(character, export_plan=plan)
        if result is None:
            raise ValueError(f"{character}: 내보낼 대표 이미지가 없습니다.")
        with result:
            return result.getvalue()
    result = io.BytesIO()
    with zipfile.ZipFile(result, "w", zipfile.ZIP_DEFLATED) as archive:
        for character, plan in plans:
            character_zip = mode.export_character_zip(character, export_plan=plan)
            if character_zip is None:
                raise ValueError(f"{character}: 내보낼 대표 이미지가 없습니다.")
            with character_zip, zipfile.ZipFile(character_zip) as source:
                for info in source.infolist():
                    target = zipfile.ZipInfo(info.filename, info.date_time)
                    target.compress_type = zipfile.ZIP_DEFLATED
                    with source.open(info) as reader, archive.open(target, "w", force_zip64=True) as writer:
                        shutil.copyfileobj(reader, writer)
    return result.getvalue()


def register_asset_export_runtime(server) -> None:
    async def export(request: web.Request) -> web.Response:
        try:
            body = await request.json()
            plans = await asyncio.to_thread(build_asset_export_plans, server.asset_mode, body)
            public = public_asset_export_plans(server.asset_mode, plans)
            if not public["success"] or request.path.endswith("/validate"):
                return web.json_response(public, status=200 if public["success"] else 409)
            # Every selected character must validate before creating any ZIP data.
            content = await asyncio.to_thread(export_asset_zip, server.asset_mode, plans)
            filename = f"{plans[0][0]}.zip" if len(plans) == 1 else "에셋 익스포트.zip"
            return web.Response(body=content, content_type="application/zip", headers={
                "Content-Disposition": f"attachment; filename*=UTF-8''{quote(filename)}",
            })
        except (ValueError, TypeError, OSError, RuntimeError) as exc:
            return web.json_response({"success": False, "error": str(exc)}, status=400)

    server.app.router.add_post("/api/asset/export/validate", export)
    server.app.router.add_post("/api/asset/export", export)
