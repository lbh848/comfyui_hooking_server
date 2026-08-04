import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class FakeCharacterMaker:
    def __init__(self):
        self.locks = {}
        self.updated = []
        self.revisions = []

    def operation_lock(self, session_id):
        return self.locks.setdefault(session_id, asyncio.Lock())

    def update_session(self, session_id, payload):
        self.updated.append((session_id, payload))
        settings = payload.get("settings") or {}
        return {
            "id": session_id,
            "settings": {
                "asset_workflow_type": "anima_only",
                "generation_workflow": settings.get("generation_workflow", "asset"),
            },
            "fields": payload.get("fields") or {},
            "llm_fields": {},
            "natural_language": payload.get("natural_language", ""),
            "llm_natural_language": "",
            "editable_preset_tags": payload.get("editable_preset_tags") or {},
            "editable_preset_enabled": payload.get("editable_preset_enabled") or {},
        }

    def save_generation_artifacts(self, session_id, **kwargs):
        self.saved_artifacts = (session_id, kwargs)
        return (
            "temporary/default/images/illustration.webp",
            "temporary/default/images/illustration_prompt.json",
        )

    def add_revision(self, session_id, **kwargs):
        self.revisions.append((session_id, kwargs))
        return {
            "id": session_id,
            "settings": {"asset_workflow_type": "anima_only"},
            "revisions": [{"id": "revision-1"}],
        }


async def _request_generation(
    server,
    monkeypatch,
    queue_result,
    *,
    generation_workflow="asset",
    image_bytes=None,
):
    captured = {}
    maker = FakeCharacterMaker()

    async def fake_add_item(item_type, label, params, **kwargs):
        captured.update(
            {
                "item_type": item_type,
                "label": label,
                "params": params,
                "kwargs": kwargs,
            }
        )
        future = asyncio.get_running_loop().create_future()
        future.set_result(queue_result)
        return SimpleNamespace(
            id="queue-character-maker",
            completion_future=future,
            generated_image_bytes=image_bytes,
        )

    async def fail_direct_generate(**_kwargs):
        raise AssertionError("캐릭터 메이커가 asset_mode.generate를 직접 호출했습니다")

    monkeypatch.setattr(server, "character_maker", maker)
    monkeypatch.setattr(server.queue_manager, "add_item", fake_add_item)
    monkeypatch.setattr(server.asset_mode, "generate", fail_direct_generate)

    app = web.Application()
    app.router.add_post(
        "/api/character_maker/session/{session_id}/generate",
        server.handle_api_character_maker_generate,
    )
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.post(
            "/api/character_maker/session/default/generate",
            json={
                "source": "user",
                "world_context": "test world",
                "fields": {
                    "appearance": ["silver_hair"],
                    "outfit": ["black_coat"],
                    "expression": [],
                    "composition": [],
                },
                "settings": {"generation_workflow": generation_workflow},
                "positive_prompt": (
                    (
                        "[FACE_ID_ACTIVATE]\ntrue\n"
                        "[FACE_ID_DIR]\nunsafe-reference\n"
                        "[END]"
                    )
                    if generation_workflow == "asset"
                    else "client-side illustration prompt is ignored"
                ),
                "negative_prompt": "low quality",
                "note": "user revision",
            },
        )
        payload = await response.json()
    finally:
        await client.close()

    return response, payload, captured, maker


@pytest.mark.asyncio
async def test_character_maker_generation_uses_integrated_asset_queue(monkeypatch):
    import server

    response, payload, captured, maker = await _request_generation(
        server,
        monkeypatch,
        {
            "success": True,
            "filename": "revision.webp",
            "local_path": "temporary/default/images/revision.webp",
            "prompt_record_path": "temporary/default/images/revision_prompt.json",
        },
    )

    assert response.status == 200
    assert payload["success"] is True
    assert payload["generation"] == {
        "success": True,
        "filename": "revision.webp",
        "generation_workflow": "asset",
    }
    assert captured["item_type"] == "asset_generation"
    assert captured["label"] == "캐릭터 메이커 사용자 이미지 생성"
    queue_body = captured["params"]["body"]
    assert queue_body["character"] == "maker-default"
    assert queue_body["asset_workflow_type"] == "anima_only"
    assert queue_body["storage_group"] == "character_maker"
    assert queue_body["storage_session"] == "default"
    assert "[FACE_ID_ACTIVATE]\nfalse" in queue_body["positive_prompt"]
    assert "[FACE_ID_DIR]\nsoya_char_ref/fallback" in queue_body["positive_prompt"]
    assert len(maker.revisions) == 1
    assert maker.revisions[0][1]["source"] == "user"


@pytest.mark.asyncio
async def test_character_maker_generation_preserves_queue_failure(monkeypatch):
    import server

    response, payload, captured, maker = await _request_generation(
        server,
        monkeypatch,
        {"success": False, "error": "ComfyUI generation failed"},
    )

    assert response.status == 500
    assert payload == {"success": False, "error": "ComfyUI generation failed"}
    assert captured["item_type"] == "asset_generation"
    assert maker.revisions == []


@pytest.mark.asyncio
async def test_character_maker_generation_routes_illustration_without_asset_tokens(
    monkeypatch,
):
    import server

    built_prompt = {
        "positive": "[ANIMA_ALL]\nsilver_hair, black_coat\n[END]",
        "negative": "low quality\n[SDXL]\nlow quality",
        "illustration_workflow_type": "v3",
        "provider": "comfy",
        "provider_mode": "comfy",
        "prompt_format": "v3",
        "width": 700,
        "height": 1024,
        "chansub_quality_tag_start": 0,
        "chansub_quality_tag_count": 0,
    }
    monkeypatch.setattr(
        server,
        "_build_character_maker_illustration_prompt",
        lambda _session, *, source: dict(built_prompt),
    )

    response, payload, captured, maker = await _request_generation(
        server,
        monkeypatch,
        {
            "success": True,
            "image_size": 3,
            "provider": "comfy",
            "illustration_workflow_type": "v3",
        },
        generation_workflow="illustration",
        image_bytes=b"img",
    )

    assert response.status == 200
    assert payload["success"] is True
    assert payload["generation"]["generation_workflow"] == "illustration"
    assert captured["item_type"] == "character_maker_illustration"
    assert captured["kwargs"]["priority"] == 0
    assert captured["params"]["positive"] == built_prompt["positive"]
    assert captured["params"]["illustration_workflow_type"] == "v3"
    assert maker.saved_artifacts[1]["image_bytes"] == b"img"
    assert maker.revisions[0][1]["positive"] == built_prompt["positive"]
