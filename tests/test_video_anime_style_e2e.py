from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from modes import video_mode as video_module
from modes.video_mode import VideoMode


@pytest.mark.asyncio
async def test_japanese_animation_plan_reaches_video_engine_prompt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Exercise V3 planning, H3 composition, and the video-engine render boundary."""

    backup_dir = tmp_path / "backups"
    raw_dir = backup_dir / "_raw"
    raw_dir.mkdir(parents=True)
    Image.new("RGB", (640, 640), (42, 64, 96)).save(
        raw_dir / "anime-source.png",
        format="PNG",
    )
    (backup_dir / "anime-source.json").write_text(
        json.dumps({"positive": "one anime girl seated beside a rainy window"}),
        encoding="utf-8",
    )
    source_ref = {"kind": "backup", "name": "anime-source"}

    calls: list[tuple[str, list[dict]]] = []
    engine_payload: dict = {}

    async def fake_vision(task_key, messages, **kwargs):
        calls.append((task_key, messages))
        metadata = kwargs.get("metadata_sink")
        if isinstance(metadata, dict):
            metadata.update(prompt_tokens=120, completion_tokens=60)
        if len(calls) == 1:
            assert task_key == "video_prompt_i2v"
            return (
                "1. 참조 연속성과 전체 방향\n"
                "창가에 앉은 인물은 편지를 읽던 키포즈를 잠시 유지한다.\n"
                "2. 타임스탬프 장면 계획\n"
                "0.00–1.80초: 화면 대부분은 정지하고 눈동자만 문장을 따라간다.\n"
                "1.80–2.10초: 짧은 선행 동작 뒤 고개를 선명하게 든다.\n"
                "2.10–5.00초: 상대를 바라보는 강한 표정과 실루엣을 홀드한다.\n"
                "3. 편집과 카메라 설계\n고정 구도를 유지한다.\n"
                "4. 시각·인물·공간 연속성\n얼굴과 의상 디자인을 고정한다.\n"
                "5. 타임라인 오디오 설계\n빗소리 속 종이 스침만 짧게 들린다.\n"
                "6. 장면별 생성 안정성 지침\n키포즈 사이 접촉과 시선을 보존한다."
            )
        assert task_key == "video_prompt_i2v"
        return (
            "visual_context:\n"
            "Picture 1: A hand-drawn anime girl sits beside a rainy window, "
            "holding a letter in both hands in a locked medium composition."
        )

    final_body = (
        "integrated_multimodal_description:\n"
        "[Shot 1] A hand-drawn Japanese-animation shot holds the girl's clear reading "
        "key pose in a locked composition. Only her pupils track the letter. After an "
        "economical anticipation, her head rises in one fast readable transition and "
        "lands on a strong expression drawing; her body and the background become still "
        "again while one delayed hair-tip accent settles.\n\n"
        "overall_soundscape:\n"
        "Steady rain, one brief paper rustle, then purposeful quiet.\n\n"
        "non_diegetic_music:\n"
        "N/A"
    )

    async def fake_text(task_key, messages, **kwargs):
        calls.append((task_key, messages))
        metadata = kwargs.get("metadata_sink")
        if isinstance(metadata, dict):
            metadata.update(prompt_tokens=180, completion_tokens=90)
        assert task_key == "video_prompt_i2v_compose"
        return final_body

    async def fake_engine_generate(payload, *, progress_callback=None):
        engine_payload.update(payload)
        assert progress_callback is None
        return b"anime-engine-mp4", {
            "execution_source": "video_engine",
            "job_id": "anime-engine-job",
            "port": 8093,
        }

    async def fake_cleanup(_descriptor, *, task_key):
        assert task_key == "video_generation"
        return True

    monkeypatch.setattr(video_module.llm_service, "callLLMVisionTask", fake_vision)
    monkeypatch.setattr(video_module.llm_service, "callLLMTask", fake_text)
    monkeypatch.setattr(
        video_module.llm_service,
        "routing_primary_model",
        lambda _task_key: "existing-llm-model",
    )
    monkeypatch.setattr(
        video_module.llm_service,
        "routing_primary_slot",
        lambda _task_key: "llm1",
    )
    monkeypatch.setattr(video_module, "_log_lighbd_history", lambda _entry: None)

    mode = VideoMode()
    mode.get_backup_dir = lambda: str(backup_dir)
    mode.get_config = lambda: {"backup_webp_quality": 80}
    mode.generate_video_engine_func = fake_engine_generate
    mode.cleanup_comfy_video_func = fake_cleanup

    plan_result = await mode.build_instruction_direct(
        {
            "mode": "i2v",
            "source_ref": source_ref,
            "instruction": "편지를 읽던 인물이 결심한 표정으로 상대를 바라본다",
            "duration": 5,
            "language": "ko",
            "refine_version": "v3",
            "include_dialogue_context": False,
            "allow_camera_motion": True,
            "allow_background_change": False,
        },
        queue_item_id="anime-plan",
    )
    prompt_params = {
        "mode": "i2v",
        "workflow_variant": "fast",
        "source_ref": source_ref,
        "instruction": plan_result["draft"],
        "instruction_original": "편지를 읽던 인물이 결심한 표정으로 상대를 바라본다",
        "llm_trace": plan_result["llm_trace"],
        "refine_version": "v3",
        "visual_context_source": "image",
        "prompt_generation_mode": "single",
        "translate_instruction_to_english": False,
        "secondary_motion": True,
        "aspect_ratio": "1:1",
        "quality_level": "medium",
        "duration": 5,
        "upscale_enabled": False,
        "output_format": "avif",
    }
    prompt_result = await mode.build_prompt(prompt_params, queue_item_id="anime-prompt")
    render_result = await mode.render_video(
        {**prompt_params, **prompt_result},
        queue_item_id="anime-render",
        use_video_engine=True,
    )

    assert plan_result["refine_version"] == "v3"
    assert plan_result["history_id"].startswith("video_instruction_direct:i2v:")
    anime_plan_messages = calls[0][1]
    assert "A held drawing is active timing" in anime_plan_messages[0]["content"]
    compose_messages = next(messages for task, messages in calls if task.endswith("_compose"))
    assert "Selected Japanese-animation generation contract" in compose_messages[0]["content"]
    assert "Do not smooth this profile back" in compose_messages[1]["content"]
    assert prompt_result["refine_version"] == "v3"
    assert engine_payload["prompt"] == prompt_result["h3_prompt"]
    assert "holds the girl's clear reading key pose" in engine_payload["prompt"]
    assert "become still again" in engine_payload["prompt"]
    assert render_result["success"] is True
    manifest = json.loads(
        (Path(render_result["postprocess_job"]["job_dir"]) / "job.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["execution_source"] == "video_engine"
