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
async def test_server_accumulates_modal_video_progress_across_samplers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    progress: list[tuple[int, int]] = []
    video_bytes = b"modal-mp4"
    artifact = {
        "remote_path": "SOYA_VIDEO_OUTPUT/video_job/result.mp4",
        "filename": "result.mp4",
        "size": len(video_bytes),
        "sha256": hashlib.sha256(video_bytes).hexdigest(),
    }

    class FakeModalService:
        async def generate_video(self, _workflow, **kwargs):
            callback = kwargs["progress_callback"]
            await callback(
                {
                    "event_type": "progress",
                    "prompt_id": "prompt-video",
                    "node": "999",
                    "value": 50,
                    "max": 100,
                }
            )
            await callback(
                {
                    "event_type": "progress",
                    "prompt_id": "prompt-video",
                    "node": "10",
                    "value": 1,
                    "max": 4,
                }
            )
            await callback(
                {
                    "event_type": "progress_state",
                    "prompt_id": "prompt-video",
                    "node": "10",
                    "value": 4,
                    "max": 4,
                }
            )
            await callback(
                {
                    "event_type": "progress",
                    "prompt_id": "prompt-video",
                    "node": "20",
                    "value": 2,
                    "max": 6,
                }
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

    # H3는 SamplerCustomAdvanced의 단계 수를 BasicScheduler에 보관한다.
    workflow = {
        "9": {"class_type": "BasicScheduler", "inputs": {"steps": 4}},
        "10": {"class_type": "SamplerCustomAdvanced", "inputs": {}},
        "19": {"class_type": "BasicScheduler", "inputs": {"steps": 6}},
        "20": {"class_type": "SamplerCustomAdvanced", "inputs": {}},
    }
    monkeypatch.setattr(server, "modal_service", FakeModalService())
    token = CURRENT_COMFY_EXECUTION_TARGET.set(MODAL_COMFY_TARGET)
    try:
        received, descriptor = await server.submit_video_workflow_to_comfy(
            workflow,
            progress_callback=on_progress,
            task_key="video_generation",
        )
    finally:
        CURRENT_COMFY_EXECUTION_TARGET.reset(token)

    assert received == video_bytes
    assert descriptor["execution_source"] == "modal"
    assert progress == [(1, 10), (4, 10), (6, 10)]


def test_modal_worker_forwards_standard_comfy_progress_events() -> None:
    worker_source = (
        Path(__file__).resolve().parents[1] / "modal_backend" / "modal_app.py"
    ).read_text(encoding="utf-8")
    frontend_source = (
        Path(__file__).resolve().parents[1] / "frontend" / "index.html"
    ).read_text(encoding="utf-8")

    assert 'event_type in ("progress", "progress_state")' in worker_source
    assert '"event_type": event_type' in worker_source
    assert "d.phase === 'h3_rendering'" in frontend_source
    assert "H3 생성 ${renderPct}%" in frontend_source


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
