from __future__ import annotations

import asyncio
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
        "instruction": "사용자가 확인한 영상 연출 지시",
        "aspect_ratio": "16:9",
        "quality_level": "high",
        "duration": 5,
        "upscale_enabled": False,
        "upscale_scale": 2,
        "output_format": "avif",
        "secondary_motion": False,
    }
    body.update(overrides)
    return body


def _draft_request(**overrides) -> dict:
    body = {
        "mode": "i2v",
        "source_ref": {"kind": "backup", "name": "source"},
        "language": "ko",
        "include_dialogue_context": False,
        "allow_camera_motion": True,
        "allow_background_change": False,
        "aspect_ratio": "16:9",
        "quality_level": "high",
        "duration": 5,
    }
    body.update(overrides)
    return body


@pytest.mark.asyncio
async def test_video_enqueue_passes_confirmed_instruction_to_prompt_queue(
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
    assert captured["params"]["instruction"] == "사용자가 확인한 영상 연출 지시"
    assert captured["params"]["auto_instruction"] is False
    assert "include_dialogue_context" not in captured["params"]
    assert captured["params"]["visual_context_source"] == "image"
    assert captured["params"]["aspect_ratio"] == "16:9"
    assert captured["params"]["quality_level"] == "high"
    assert captured["params"]["secondary_motion"] is False


@pytest.mark.asyncio
async def test_video_enqueue_rejects_non_boolean_secondary_motion(monkeypatch) -> None:
    called = False

    async def fake_add_item(_item_type, _label, _params):
        nonlocal called
        called = True
        raise AssertionError("invalid request must not be queued")

    monkeypatch.setattr(server.queue_manager, "add_item", fake_add_item)

    response = await server.handle_api_video_enqueue(
        _JsonRequest(_video_request(secondary_motion="false"))
    )
    payload = json.loads(response.text)

    assert response.status == 400
    assert payload["success"] is False
    assert "boolean" in payload["error"]
    assert called is False


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
async def test_fast_video_enqueue_forces_native_768p_profile(monkeypatch) -> None:
    captured: dict = {}

    async def fake_add_item(item_type, label, params):
        captured.update(item_type=item_type, label=label, params=params)
        return SimpleNamespace(id="fast-video-prompt", label=label)

    monkeypatch.setattr(server.video_mode, "validate_reference", lambda _reference: None)
    monkeypatch.setattr(server.queue_manager, "add_item", fake_add_item)
    monkeypatch.setattr(server, "load_config", lambda: {})

    response = await server.handle_api_video_enqueue(
        _JsonRequest(
            _video_request(
                workflow_variant="fast",
                aspect_ratio="16:9",
                quality_level="ignored-by-fast-profile",
            )
        )
    )

    assert response.status == 200
    assert captured["params"]["workflow_variant"] == "fast"
    assert captured["params"]["aspect_ratio"] == "16:9"
    assert captured["params"]["quality_level"] == "native"
    assert captured["label"] == "H3 고속 I2V 프롬프트"


@pytest.mark.asyncio
async def test_fast_video_enqueue_rejects_unsupported_ultrawide_ratio(
    monkeypatch,
) -> None:
    called = False

    async def fake_add_item(_item_type, _label, _params):
        nonlocal called
        called = True
        raise AssertionError("invalid request must not be queued")

    monkeypatch.setattr(server.queue_manager, "add_item", fake_add_item)

    response = await server.handle_api_video_enqueue(
        _JsonRequest(
            _video_request(workflow_variant="fast", aspect_ratio="21:9")
        )
    )
    payload = json.loads(response.text)

    assert response.status == 400
    assert payload["success"] is False
    assert "화면 비율" in payload["error"]
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
async def test_video_enqueue_rejects_removed_integrated_auto_instruction(
    monkeypatch,
) -> None:
    called = False

    async def fake_add_item(_item_type, _label, _params):
        nonlocal called
        called = True
        raise AssertionError("invalid request must not be queued")

    monkeypatch.setattr(server.queue_manager, "add_item", fake_add_item)

    response = await server.handle_api_video_enqueue(
        _JsonRequest(_video_request(auto_instruction=True))
    )
    payload = json.loads(response.text)

    assert response.status == 400
    assert payload["success"] is False
    assert "AI 초안 만들기" in payload["error"]
    assert called is False


@pytest.mark.asyncio
async def test_video_instruction_draft_waits_for_llm_queue_and_passes_options(
    monkeypatch,
) -> None:
    captured: dict = {}

    async def fake_add_item(item_type, label, params):
        captured.update(item_type=item_type, label=label, params=params)
        future = asyncio.get_running_loop().create_future()
        future.set_result(
            {
                "success": True,
                "draft": "인물이 천천히 고개를 든다.",
                "language": "ko",
                "history_id": "draft-history",
            }
        )
        return SimpleNamespace(
            id="video-draft-id",
            label=label,
            completion_future=future,
        )

    monkeypatch.setattr(server.video_mode, "validate_reference", lambda _reference: None)
    monkeypatch.setattr(server.queue_manager, "add_item", fake_add_item)

    response = await server.handle_api_video_instruction_draft(
        _JsonRequest(_draft_request(allow_background_change=True))
    )
    payload = json.loads(response.text)

    assert response.status == 200
    assert payload["success"] is True
    assert payload["draft"] == "인물이 천천히 고개를 든다."
    assert captured["item_type"] == "video_instruction_draft"
    assert captured["params"]["language"] == "ko"
    assert captured["params"]["include_dialogue_context"] is False
    assert captured["params"]["allow_camera_motion"] is True
    assert captured["params"]["allow_background_change"] is True


@pytest.mark.asyncio
async def test_video_instruction_draft_rejects_invalid_option_before_queue(
    monkeypatch,
) -> None:
    called = False

    async def fake_add_item(_item_type, _label, _params):
        nonlocal called
        called = True
        raise AssertionError("invalid request must not be queued")

    monkeypatch.setattr(server.queue_manager, "add_item", fake_add_item)

    response = await server.handle_api_video_instruction_draft(
        _JsonRequest(_draft_request(allow_camera_motion="yes"))
    )
    payload = json.loads(response.text)

    assert response.status == 400
    assert payload["success"] is False
    assert "boolean" in payload["error"]
    assert called is False


@pytest.mark.asyncio
async def test_video_reprocess_enqueue_stages_existing_animation(monkeypatch) -> None:
    captured: dict = {}

    def fake_stage(params):
        captured["stage_params"] = params
        return {
            "job_dir": "spool/job",
            "job_kind": "existing_animation",
            "spool_id": "reprocess-1",
            "mode": "reprocess",
            "source_label": "source",
            "upscale_enabled": False,
            "upscale_scale": 2,
            "upscale_model": "",
            "output_format": "webp",
            "fps": 12,
            "target_size_bytes": 2 * 1024 * 1024,
        }

    async def fake_add_item(item_type, label, params):
        captured.update(item_type=item_type, label=label, queue_params=params)
        return SimpleNamespace(id="postprocess-id", label=label)

    monkeypatch.setattr(server.video_mode, "stage_existing_animation_postprocess", fake_stage)
    monkeypatch.setattr(server.queue_manager, "add_item", fake_add_item)
    monkeypatch.setattr(server, "load_config", lambda: {})

    response = await server.handle_api_video_reprocess_enqueue(
        _JsonRequest(
            {
                "source_ref": {"kind": "backup", "name": "source"},
                "target_size_mb": 2,
                "fps": 12,
                "upscale_enabled": False,
                "upscale_scale": 2,
                "output_format": "webp",
            }
        )
    )
    payload = json.loads(response.text)

    assert response.status == 200
    assert payload["success"] is True
    assert captured["item_type"] == server.VIDEO_POSTPROCESS_TYPE
    assert captured["stage_params"]["fps"] == 12
    assert captured["stage_params"]["target_size_mb"] == 2
    assert captured["queue_params"]["job_kind"] == "existing_animation"


@pytest.mark.asyncio
async def test_video_reprocess_enqueue_rejects_invalid_fps_before_staging(
    monkeypatch,
) -> None:
    called = False

    def fake_stage(_params):
        nonlocal called
        called = True
        raise AssertionError("invalid request must not be staged")

    monkeypatch.setattr(server.video_mode, "stage_existing_animation_postprocess", fake_stage)
    response = await server.handle_api_video_reprocess_enqueue(
        _JsonRequest(
            {
                "source_ref": {"kind": "backup", "name": "source"},
                "target_size_mb": 2,
                "fps": 61,
                "upscale_enabled": False,
                "upscale_scale": 2,
                "output_format": "webp",
            }
        )
    )
    payload = json.loads(response.text)

    assert response.status == 400
    assert payload["success"] is False
    assert "FPS" in payload["error"]
    assert called is False
