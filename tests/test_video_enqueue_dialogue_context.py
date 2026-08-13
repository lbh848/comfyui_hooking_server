from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import server


class _JsonRequest:
    def __init__(self, body: dict) -> None:
        self._body = body

    async def json(self) -> dict:
        return self._body


def _video_request(**overrides) -> dict:
    body = {
        "mode": "i2v",
        "source_ref": {"kind": "backup", "name": "source"},
        "auto_instruction": True,
        "include_dialogue_context": False,
        "aspect_ratio": "16:9",
        "quality_level": "high",
        "duration": 5,
        "upscale_enabled": False,
        "upscale_scale": 2,
        "output_format": "avif",
    }
    body.update(overrides)
    return body


@pytest.mark.asyncio
async def test_video_enqueue_passes_dialogue_context_choice_to_prompt_queue(
    monkeypatch,
) -> None:
    captured: dict = {}

    async def fake_add_item(item_type, label, params):
        captured.update(item_type=item_type, label=label, params=params)
        return SimpleNamespace(id="video-prompt-id", label=label)

    monkeypatch.setattr(server.video_mode, "validate_reference", lambda _reference: None)
    monkeypatch.setattr(server.queue_manager, "add_item", fake_add_item)
    monkeypatch.setattr(server, "load_config", lambda: {})

    response = await server.handle_api_video_enqueue(_JsonRequest(_video_request()))
    payload = json.loads(response.text)

    assert response.status == 200
    assert payload["success"] is True
    assert captured["item_type"] == "video_prompt_build"
    assert captured["params"]["include_dialogue_context"] is False
    assert captured["params"]["visual_context_source"] == "image"
    assert captured["params"]["aspect_ratio"] == "16:9"
    assert captured["params"]["quality_level"] == "high"


@pytest.mark.asyncio
async def test_video_enqueue_passes_prompt_visual_context_choice(monkeypatch) -> None:
    captured: dict = {}

    async def fake_add_item(item_type, label, params):
        captured.update(item_type=item_type, label=label, params=params)
        return SimpleNamespace(id="prompt-context-id", label=label)

    monkeypatch.setattr(server.video_mode, "validate_reference", lambda _reference: None)
    monkeypatch.setattr(server.queue_manager, "add_item", fake_add_item)
    monkeypatch.setattr(server, "load_config", lambda: {})

    response = await server.handle_api_video_enqueue(
        _JsonRequest(_video_request(visual_context_source="prompt"))
    )

    assert response.status == 200
    assert captured["params"]["visual_context_source"] == "prompt"


@pytest.mark.asyncio
async def test_video_enqueue_rejects_unknown_visual_context_source(monkeypatch) -> None:
    called = False

    async def fake_add_item(_item_type, _label, _params):
        nonlocal called
        called = True
        raise AssertionError("invalid request must not be queued")

    monkeypatch.setattr(server.queue_manager, "add_item", fake_add_item)

    response = await server.handle_api_video_enqueue(
        _JsonRequest(_video_request(visual_context_source="metadata"))
    )
    payload = json.loads(response.text)

    assert response.status == 400
    assert payload["success"] is False
    assert "image 또는 prompt" in payload["error"]
    assert called is False


@pytest.mark.asyncio
async def test_video_enqueue_rejects_unknown_fast_quality_level(monkeypatch) -> None:
    called = False

    async def fake_add_item(_item_type, _label, _params):
        nonlocal called
        called = True
        raise AssertionError("invalid request must not be queued")

    monkeypatch.setattr(server.video_mode, "validate_reference", lambda _reference: None)
    monkeypatch.setattr(server.queue_manager, "add_item", fake_add_item)
    monkeypatch.setattr(server, "load_config", lambda: {})

    response = await server.handle_api_video_enqueue(
        _JsonRequest(_video_request(quality_level="ultra"))
    )
    payload = json.loads(response.text)

    assert response.status == 400
    assert payload["success"] is False
    assert "화질 단계" in payload["error"]
    assert called is False


@pytest.mark.asyncio
async def test_video_enqueue_keeps_legacy_preset_requests_compatible(monkeypatch) -> None:
    captured: dict = {}

    async def fake_add_item(item_type, label, params):
        captured.update(item_type=item_type, label=label, params=params)
        return SimpleNamespace(id="legacy-video-prompt", label=label)

    monkeypatch.setattr(server.video_mode, "validate_reference", lambda _reference: None)
    monkeypatch.setattr(server.queue_manager, "add_item", fake_add_item)
    monkeypatch.setattr(server, "load_config", lambda: {})
    body = _video_request()
    body.pop("aspect_ratio")
    body.pop("quality_level")
    body["preset"] = "4:3"

    response = await server.handle_api_video_enqueue(_JsonRequest(body))

    assert response.status == 200
    assert captured["params"]["aspect_ratio"] == "4:3"
    assert captured["params"]["quality_level"] == "medium"


@pytest.mark.asyncio
async def test_video_enqueue_rejects_non_boolean_dialogue_context_choice(
    monkeypatch,
) -> None:
    called = False

    async def fake_add_item(_item_type, _label, _params):
        nonlocal called
        called = True
        raise AssertionError("invalid request must not be queued")

    monkeypatch.setattr(server.queue_manager, "add_item", fake_add_item)

    response = await server.handle_api_video_enqueue(
        _JsonRequest(_video_request(include_dialogue_context="false"))
    )
    payload = json.loads(response.text)

    assert response.status == 400
    assert payload["success"] is False
    assert "boolean" in payload["error"]
    assert called is False
