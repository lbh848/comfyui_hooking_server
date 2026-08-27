from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest
from PIL import Image

from modes.video_mode import VideoMode


@pytest.mark.asyncio
async def test_ref2v_video_engine_skips_comfy_and_sends_three_original_refs(
    tmp_path: Path,
) -> None:
    backup_dir = tmp_path / "backups"
    raw_dir = backup_dir / "_raw"
    raw_dir.mkdir(parents=True)
    refs = []
    for index, size in enumerate(((800, 600), (400, 900), (1200, 500)), start=1):
        name = f"ref-{index}"
        Image.new("RGB", size, (index * 40, 80, 120)).save(
            raw_dir / f"{name}.png",
            format="PNG",
        )
        (backup_dir / f"{name}.json").write_text(
            json.dumps({"positive": f"reference {index}"}),
            encoding="utf-8",
        )
        refs.append({"kind": "backup", "name": name})

    captured: dict = {}
    cleaned: list[dict] = []

    async def generate(payload, *, progress_callback=None):
        captured.update(payload)
        assert progress_callback is None
        assert payload["mode"] == "ref2v"
        assert payload["width"] == 864
        assert payload["height"] == 384
        assert payload["frames"] == 124
        assert payload["fps"] == 24
        assert "first_frame" not in payload
        assert "last_frame" not in payload
        assert len(payload["reference_images"]) == 3
        for encoded in payload["reference_images"]:
            prefix, value = encoded.split(",", 1)
            assert prefix == "data:image/png;base64"
            assert base64.b64decode(value).startswith(b"\x89PNG")
        return b"engine-mp4", {
            "execution_source": "video_engine",
            "job_id": "engine-job-1",
            "port": 8093,
        }

    async def cleanup(descriptor, *, task_key):
        assert task_key == "video_generation"
        assert descriptor["job_id"] == "engine-job-1"
        cleaned.append(descriptor)
        return True

    mode = VideoMode()
    mode.get_backup_dir = lambda: str(backup_dir)
    # comfy_input_dir와 워크플로우 경로가 없어도 외부 엔진 분기는 동작해야 한다.
    mode.get_config = lambda: {"backup_webp_quality": 80}
    mode.generate_video_engine_func = generate
    mode.cleanup_comfy_video_func = cleanup
    mode.convert_workflow_func = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("외부 엔진에서 Comfy 워크플로우 변환을 호출하면 안 됩니다")
    )
    mode.submit_workflow_func = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("외부 엔진에서 Comfy 제출을 호출하면 안 됩니다")
    )

    result = await mode.render_video(
        {
            "mode": "ref2v",
            "workflow_variant": "fast",
            "source_ref": refs[0],
            "reference_refs": refs,
            "instruction": "세 인물이 함께 걷는다",
            "instruction_source": "user",
            "visual_context": "visual_context:\nThree independent references.",
            "aspect_ratio": "21:9",
            "quality_level": "medium",
            "duration": 5,
            "h3_prompt": (
                "A complete reference-video direction written as natural language "
                "without requiring the supplied pictures to be timeline keyframes."
            ),
        },
        queue_item_id="ref-engine",
        use_video_engine=True,
    )

    assert result["success"] is True
    assert result["width"] == 864
    assert result["height"] == 384
    assert captured["seed"] >= 0
    assert cleaned and cleaned[0]["execution_source"] == "video_engine"
    manifest = json.loads(
        (Path(result["postprocess_job"]["job_dir"]) / "job.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["execution_source"] == "video_engine"
    assert manifest["reference_refs"] == refs
