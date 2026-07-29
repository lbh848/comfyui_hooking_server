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
        return {
            "id": session_id,
            "settings": {"asset_workflow_type": "anima_only"},
        }

    def add_revision(self, session_id, **kwargs):
        self.revisions.append((session_id, kwargs))
        return {
            "id": session_id,
            "settings": {"asset_workflow_type": "anima_only"},
            "revisions": [{"id": "revision-1"}],
        }


async def _request_generation(server, monkeypatch, queue_result):
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
        return SimpleNamespace(id="queue-character-maker", completion_future=future)

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
                "positive_prompt": (
                    "[FACE_ID_ACTIVATE]\ntrue\n"
                    "[FACE_ID_DIR]\nunsafe-reference\n"
                    "[END]"
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
