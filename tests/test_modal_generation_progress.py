from __future__ import annotations

import pytest

import server


@pytest.mark.asyncio
async def test_modal_illustration_streams_sampler_progress_to_queue_and_frontend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue_progress: list[tuple[int | float, int | float]] = []
    frontend_events: list[tuple[str, dict]] = []

    async def update_workflow(_workflow_type: str) -> None:
        return None

    async def notify(event_type: str, data: dict) -> None:
        frontend_events.append((event_type, dict(data)))

    async def generate(_workflow: dict, **kwargs):
        await kwargs["progress_callback"](
            {
                "value": 6,
                "max": 20,
                "node": "sampler-1",
                "event_type": "progress",
            }
        )
        return b"modal-image", {"prompt_id": "modal-illustration"}

    async def on_queue_progress(value, max_value) -> None:
        queue_progress.append((value, max_value))

    monkeypatch.setattr(server, "update_workflow_if_needed", update_workflow)
    monkeypatch.setattr(
        server,
        "build_prompt",
        lambda _positive, _negative: {
            "1": {"class_type": "KSampler", "inputs": {"steps": 20}}
        },
    )
    monkeypatch.setattr(server, "notify_frontend", notify)
    monkeypatch.setattr(server.modal_service, "generate", generate)
    monkeypatch.setattr(server, "current_original_workflow", {})
    monkeypatch.setattr(
        server,
        "current_api_workflow",
        {"1": {"class_type": "KSampler", "inputs": {"steps": 20}}},
    )
    monkeypatch.setattr(server, "current_conversion_info", {})

    token = server.CURRENT_COMFY_EXECUTION_TARGET.set(server.MODAL_COMFY_TARGET)
    try:
        image, metadata = await server.generate_image_with_prompt(
            "positive",
            "negative",
            progress_callback=on_queue_progress,
            provider="comfy",
        )
    finally:
        server.CURRENT_COMFY_EXECUTION_TARGET.reset(token)

    assert image == b"modal-image"
    assert metadata["modal"]["prompt_id"] == "modal-illustration"
    assert queue_progress == [(0, 1), (6, 20), (1, 1)]
    assert frontend_events == [
        ("generation_progress", {"value": 0, "max": 1}),
        ("generation_progress", {"value": 6, "max": 20}),
        ("generation_progress", {"value": 1, "max": 1}),
    ]


@pytest.mark.asyncio
async def test_modal_asset_streams_percentage_progress_to_existing_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    progress: list[tuple[int | float, int | float]] = []
    captured: dict = {}

    async def generate(_workflow: dict, **kwargs):
        captured.update(kwargs)
        await kwargs["progress_callback"](
            {"phase": "modal_syncing", "percentage": 35.5}
        )
        return b"modal-asset", {"prompt_id": "modal-asset-prompt"}

    async def on_progress(value, max_value) -> None:
        progress.append((value, max_value))

    monkeypatch.setattr(server.modal_service, "generate", generate)
    token = server.CURRENT_COMFY_EXECUTION_TARGET.set(server.MODAL_COMFY_TARGET)
    try:
        image, metadata = await server.submit_workflow_to_comfy(
            {"1": {"class_type": "SaveImage", "inputs": {}}},
            progress_callback=on_progress,
            input_paths=["input/reference"],
        )
    finally:
        server.CURRENT_COMFY_EXECUTION_TARGET.reset(token)

    assert image == b"modal-asset"
    assert metadata["modal"]["prompt_id"] == "modal-asset-prompt"
    assert captured["input_paths"] == ["input/reference"]
    assert progress == [(0, 1), (35.5, 100), (1, 1)]
