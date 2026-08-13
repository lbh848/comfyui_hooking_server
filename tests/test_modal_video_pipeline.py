from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from comfy_allocation import CURRENT_COMFY_EXECUTION_TARGET, MODAL_COMFY_TARGET
import server


@pytest.mark.asyncio
async def test_server_routes_video_to_modal_and_forwards_staged_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_dir = tmp_path / "soya_video"
    input_dir.mkdir()
    first = input_dir / "[1].png"
    first.write_bytes(b"image")
    observed: dict = {}
    progress: list[tuple[int, int]] = []
    video_bytes = b"modal-mp4"
    artifact = {
        "remote_path": "SOYA_VIDEO_OUTPUT/video_job/result.mp4",
        "filename": "result.mp4",
        "size": len(video_bytes),
        "sha256": hashlib.sha256(video_bytes).hexdigest(),
    }

    class FakeModalService:
        async def generate_video(self, workflow, **kwargs):
            observed["workflow"] = workflow
            observed.update(kwargs)
            await kwargs["progress_callback"](
                {"sequence": 1, "data": {"step": 2, "total": 4}}
            )
            return video_bytes, {
                "execution_source": "modal",
                "prompt_id": "prompt-video",
                "filename": "result.mp4",
                "type": "output",
                "artifact": artifact,
            }

    async def on_progress(value: int, maximum: int) -> None:
        progress.append((value, maximum))

    monkeypatch.setattr(server, "modal_service", FakeModalService())
    monkeypatch.setattr(
        server,
        "resolve_comfy_port",
        lambda _task: (_ for _ in ()).throw(
            AssertionError("Modal 영상이 로컬 Comfy 포트를 조회하면 안 됩니다.")
        ),
    )
    token = CURRENT_COMFY_EXECUTION_TARGET.set(MODAL_COMFY_TARGET)
    try:
        received, descriptor = await server.submit_video_workflow_to_comfy(
            {"1": {"class_type": "Test", "inputs": {}}},
            progress_callback=on_progress,
            task_key="video_generation",
            input_paths=[str(input_dir)],
        )
    finally:
        CURRENT_COMFY_EXECUTION_TARGET.reset(token)

    assert received == video_bytes
    assert descriptor["execution_source"] == "modal"
    assert observed["input_paths"] == [str(input_dir)]
    assert progress == [(2, 4)]


@pytest.mark.asyncio
async def test_server_modal_cleanup_uses_verified_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = {
        "remote_path": "SOYA_VIDEO_OUTPUT/video_job/result.mp4",
        "filename": "result.mp4",
        "size": 6,
        "sha256": hashlib.sha256(b"abcdef").hexdigest(),
    }
    observed: list[list[dict]] = []

    class FakeModalService:
        async def delete_video_artifacts_after_spool(self, artifacts):
            observed.append(artifacts)
            return True

    monkeypatch.setattr(server, "modal_service", FakeModalService())

    cleaned = await server.cleanup_comfy_video_output(
        {
            "execution_source": "modal",
            "filename": "result.mp4",
            "type": "output",
            "artifact": artifact,
        },
        task_key="video_generation",
    )

    assert cleaned is True
    assert observed == [[artifact]]
