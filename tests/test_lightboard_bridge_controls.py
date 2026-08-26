import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from aiohttp import web

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import server
from modes import illustration_context_pipeline as pipeline


class _JsonRequest:
    def __init__(self, method="GET", body=None):
        self.method = method
        self._body = body or {}

    async def json(self):
        return self._body


@pytest.mark.asyncio
async def test_bridge_health_advertises_media_display_metadata():
    response = await server.handle_api_illustration_context_bridge_health(None)
    payload = json.loads(response.text)
    assert payload["version"] == 10
    assert payload["bot_selection"] is True
    assert payload["easy_edit"] is True
    assert payload["slot_animation_metadata"] is True
    assert payload["asset_display_metadata"] is True
    assert payload["asset_reroll"] is True
    assert "asset_reroll" in payload["progress_phases"]


@pytest.mark.asyncio
async def test_easy_edit_queue_rejects_animated_slot(monkeypatch, tmp_path):
    session_id = "animated_easy_edit_session_123456"
    prompt_id = "animated-easy-edit-prompt"
    monkeypatch.setattr(pipeline, "SESSION_DIR", str(tmp_path / "sessions"))
    pipeline.create_session(session_id, "context")
    pipeline.set_session_result(
        session_id,
        [{
            "slot": 2,
            "raw_positive": "animated",
            "raw_negative": "",
            "backup_name": "animated-source",
            "animated": True,
        }],
        [b"animated-image"],
    )
    item = SimpleNamespace(params={
        "prompt_id": prompt_id,
        "payload": {
            "session_id": session_id,
            "slot": 2,
            "direction": "배경을 바꿔줘",
        },
    })

    with pytest.raises(RuntimeError, match="애니메이션 삽화"):
        await server.process_illustration_easy_edit_queue_item(item)

    pipeline._SESSIONS.pop(session_id, None)


@pytest.mark.asyncio
async def test_bridge_bot_selector_returns_names_and_saves_valid_selection(monkeypatch):
    config = {"bot_selected": "Bot A", "bot_mode_enabled": True}
    saved = []
    monkeypatch.setattr(server, "app_config", dict(config))
    monkeypatch.setattr(
        server,
        "_load_bot_data_readonly",
        lambda: {"bots": [{"name": "Bot A"}, {"name": "Bot B"}]},
    )
    monkeypatch.setattr(server, "load_config", lambda: dict(config))
    monkeypatch.setattr(server, "save_config", lambda value: saved.append(dict(value)))

    response = await server.handle_api_illustration_context_bridge_bots(
        _JsonRequest("GET")
    )
    assert json.loads(response.text) == {
        "ok": True,
        "bots": ["Bot A", "Bot B"],
        "bot_selected": "Bot A",
    }

    response = await server.handle_api_illustration_context_bridge_bots(
        _JsonRequest("POST", {"bot_selected": "Bot B"})
    )
    assert response.status == 200
    assert json.loads(response.text)["bot_selected"] == "Bot B"
    assert saved[-1]["bot_selected"] == "Bot B"


@pytest.mark.asyncio
async def test_bridge_bot_selector_rejects_unknown_selection(monkeypatch):
    monkeypatch.setattr(server, "app_config", {"bot_selected": "Bot A"})
    monkeypatch.setattr(
        server,
        "_load_bot_data_readonly",
        lambda: {"bots": [{"name": "Bot A"}]},
    )
    monkeypatch.setattr(server, "load_config", lambda: {"bot_selected": "Bot A"})
    saved = []
    monkeypatch.setattr(server, "save_config", lambda value: saved.append(value))

    response = await server.handle_api_illustration_context_bridge_bots(
        _JsonRequest("POST", {"bot_selected": "missing"})
    )
    assert response.status == 400
    assert saved == []


@pytest.mark.asyncio
async def test_easy_edit_bridge_reuses_existing_edit_and_regenerate_handlers(
    monkeypatch,
    tmp_path,
):
    session_id = "easy_edit_session_123456"
    prompt_id = "easy-edit-prompt"
    monkeypatch.setattr(pipeline, "SESSION_DIR", str(tmp_path / "sessions"))
    pipeline.create_session(session_id, "context")
    pipeline.set_session_result(
        session_id,
        [{
            "slot": 2,
            "raw_positive": "[NAME]\nhero",
            "raw_negative": "",
            "backup_name": "source-backup",
        }],
        [b"old-image"],
    )
    (tmp_path / "source-backup.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(server, "WORKFLOW_BACKUP_DIR", str(tmp_path))
    monkeypatch.setattr(
        server,
        "_extract_prompts_from_backup",
        lambda path: ("source positive", "source negative"),
    )
    monkeypatch.setattr(
        server,
        "_llm_edit_identity_capability",
        lambda backup_name: {
            "enabled": True,
            "reason": "",
            "character_names": ["hero"],
            "visual_profile_ids": {"hero": "awakened"},
        },
    )

    edit_body = {}
    regenerate_body = {}

    async def fake_edit(request, *, _body=None):
        edit_body.update(_body)
        return web.json_response({
            "plan": "night",
            "positive": "edited positive",
            "negative": "edited negative",
        })

    async def fake_regenerate(
        request,
        *,
        _body=None,
        _return_queue_result=False,
    ):
        regenerate_body.update(_body)
        assert _return_queue_result is True
        return {
            "success": True,
            "image_bytes": b"edited-image",
            "backup_name": "edited-backup",
        }

    completed = []

    async def fake_complete(actual_prompt_id, save_node, filename):
        completed.append((actual_prompt_id, save_node, filename))
        server.prompts[actual_prompt_id]["status"] = "completed"

    monkeypatch.setattr(server, "handle_api_llm_edit_prompt", fake_edit)
    monkeypatch.setattr(
        server,
        "handle_api_reschedule_with_modified_prompt",
        fake_regenerate,
    )
    monkeypatch.setattr(server, "complete_prompt_from_reschedule", fake_complete)
    server.prompts[prompt_id] = {
        "status": "running",
        "save_node_id": "9",
        "image_bytes": None,
    }
    item = SimpleNamespace(params={
        "prompt_id": prompt_id,
        "payload": {
            "session_id": session_id,
            "slot": 2,
            "direction": "배경을 밤으로 바꿔줘",
        },
    })

    result = await server.process_illustration_easy_edit_queue_item(item)

    assert result["success"] is True
    assert edit_body["name"] == "source-backup"
    assert edit_body["direction"] == "배경을 밤으로 바꿔줘"
    assert edit_body["characters"] == ["hero"]
    assert edit_body["visual_profile_ids"] == {"hero": "awakened"}
    assert regenerate_body["positive"] == "edited positive"
    assert server.prompts[prompt_id]["image_bytes"] == b"edited-image"
    assert pipeline.session_image_by_slot(session_id, 2) == b"edited-image"
    assert pipeline.session_item_by_slot(session_id, 2)["backup_name"] == "edited-backup"
    assert completed and completed[0][0] == prompt_id
