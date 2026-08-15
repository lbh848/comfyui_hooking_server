import io
import json
import sys
from pathlib import Path

import pillow_avif  # noqa: F401  # Pillow AVIF codec registration
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import server
from modes import illustration_context_pipeline as pipeline


class _JsonRequest:
    method = "POST"

    def __init__(self, body: dict):
        self._body = body

    async def json(self) -> dict:
        return self._body


class _MatchRequest:
    def __init__(self, match_info: dict):
        self.match_info = match_info


def _build_two_frame_animation(image_format: str) -> bytes:
    frames = [
        Image.new("RGB", (16, 16), color=(255, 128, 0)),
        Image.new("RGB", (16, 16), color=(0, 128, 255)),
    ]
    output = io.BytesIO()
    frames[0].save(
        output,
        format=image_format,
        save_all=True,
        append_images=frames[1:],
        duration=200,
        loop=0,
    )
    return output.getvalue()


@pytest.mark.asyncio
async def test_reschedule_cancel_does_not_require_backup_name(monkeypatch) -> None:
    notifications = []

    async def fake_notify(event_type: str, data: dict) -> None:
        notifications.append((event_type, data))

    monkeypatch.setattr(server, "notify_frontend", fake_notify)
    monkeypatch.setattr(server, "reschedule_queue", {"name": "old-backup"})

    response = await server.handle_api_reschedule(_JsonRequest({"action": "cancel"}))

    assert response.status == 200
    assert server.reschedule_queue is None
    assert notifications == [
        ("reschedule_changed", {"scheduled": False, "name": None})
    ]


@pytest.mark.asyncio
async def test_reschedule_loads_animated_avif_from_configured_backup_dir(
    tmp_path,
    monkeypatch,
) -> None:
    backup_name = "animated-backup"
    animation_bytes = _build_two_frame_animation("AVIF")
    (tmp_path / f"{backup_name}.avif").write_bytes(animation_bytes)
    (tmp_path / f"{backup_name}.json").write_text(
        json.dumps(
            {
                "provider": "video",
                "positive": "animated positive",
                "negative": "animated negative",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    notifications = []

    async def fake_notify(event_type: str, data: dict) -> None:
        notifications.append((event_type, data))

    monkeypatch.setattr(server, "get_backup_base_dir", lambda: str(tmp_path))
    monkeypatch.setattr(server, "notify_frontend", fake_notify)
    monkeypatch.setattr(server, "reschedule_queue", None)

    response = await server.handle_api_reschedule(
        _JsonRequest({"name": backup_name, "action": "toggle"})
    )
    payload = json.loads(response.text)

    assert response.status == 200
    assert payload == {
        "scheduled": True,
        "name": backup_name,
        "message": "Backup scheduled for retransmission",
    }
    assert server.reschedule_queue["image_bytes"] == animation_bytes
    assert server.reschedule_queue["positive"] == "animated positive"
    assert server.reschedule_queue["negative"] == "animated negative"
    assert notifications == [
        ("reschedule_changed", {"scheduled": True, "name": backup_name})
    ]

    session_id = "risu_" + ("a" * 64)
    slot = 3
    monkeypatch.setattr(pipeline, "SESSION_DIR", str(tmp_path / "sessions"))
    monkeypatch.setattr(server, "prompts", {})
    pipeline._SESSIONS.pop(session_id, None)
    pipeline.create_session(session_id, "animated reservation")
    pipeline.set_session_result(
        session_id,
        [{"slot": slot, "raw_positive": "scene", "raw_negative": ""}],
        [b"old-static-placeholder"],
    )

    async def fake_complete_prompt(*args, **kwargs) -> None:
        return None

    monkeypatch.setattr(server, "complete_prompt_from_reschedule", fake_complete_prompt)
    consume_response = await server._serve_priority_reservation_for_illustration_slot(
        {"client_id": "test-client", "extra_data": {}},
        "animated-reservation-prompt",
        {},
        session_id,
        slot,
        "scene",
    )
    bridge_response = await server.handle_api_illustration_context_bridge_image(
        _MatchRequest({"sid": session_id, "slot": str(slot)})
    )

    assert consume_response is not None
    assert consume_response.status == 200
    assert server.reschedule_queue is None
    assert bridge_response.status == 200
    assert bridge_response.content_type == "image/avif"
    assert bridge_response.body == animation_bytes
    descriptor = pipeline.session_item_by_slot(session_id, slot)
    assert descriptor["backup_name"] == backup_name
    assert descriptor["animated"] is True
