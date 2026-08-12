from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

import modes.video_mode as video_module
from modes.video_mode import (
    FAST_PRESETS,
    FIRST_LAST_ALIGNMENT,
    I2V_ALIGNMENT,
    VideoMode,
    build_i2v_workflow_block,
    center_crop_to_ratio,
    choose_fast_preset,
    compose_h3_prompt,
    normalize_h3_prompt_body,
    normalize_visual_context,
    validate_h3_prompt,
    validate_h3_prompt_body,
    validate_visual_context,
)


ROOT = Path(__file__).resolve().parents[1]


def _valid_body(prefix: str = "") -> str:
    return (
        prefix
        + ("\n\n" if prefix else "")
        + "integrated_multimodal_description:\n"
        + "[Shot 1] A subject moves continuously while the camera slowly pushes in slightly.\n\n"
        + "overall_soundscape:\nA quiet room with subtle movement sounds.\n\n"
        + "non_diegetic_music:\nNo music."
    )


def test_fast_presets_match_product_contract() -> None:
    assert FAST_PRESETS == {
        "1:1": (512, 512),
        "4:3": (512, 384),
        "3:4": (384, 512),
        "16:9": (672, 384),
        "9:16": (384, 672),
        "21:9": (672, 288),
        "9:21": (288, 672),
        "3:2": (576, 384),
        "2:3": (384, 576),
        "5:4": (480, 384),
        "4:5": (384, 480),
    }
    assert choose_fast_preset(1536, 1536) == "1:1"
    assert choose_fast_preset(1536, 864) == "16:9"
    assert choose_fast_preset(864, 1536) == "9:16"


def test_crop_happens_at_source_resolution_before_fast_resize() -> None:
    source = Image.new("RGB", (1536, 1024), "red")

    cropped = center_crop_to_ratio(source, 512, 512)

    assert cropped.size == (1024, 1024)
    assert cropped.size != FAST_PRESETS["1:1"]
    assert cropped.resize(FAST_PRESETS["1:1"], Image.Resampling.LANCZOS).size == (
        512,
        512,
    )


def test_h3_prompt_validator_enforces_each_official_mode_header() -> None:
    assert validate_h3_prompt(_valid_body(), "t2v") == (True, "")
    assert validate_h3_prompt(_valid_body(I2V_ALIGNMENT), "i2v") == (True, "")
    assert validate_h3_prompt(_valid_body(FIRST_LAST_ALIGNMENT), "first_last") == (
        True,
        "",
    )
    assert validate_h3_prompt(_valid_body(), "i2v")[0] is False
    assert validate_h3_prompt(_valid_body(I2V_ALIGNMENT), "t2v")[0] is False
    assert validate_h3_prompt(
        "integrated_multimodal_description:\n[Shot 1] X\n\n"
        "non_diegetic_music:\nNone\n\n"
        "overall_soundscape:\nQuiet",
        "t2v",
    )[0] is False


def test_program_adds_exact_i2v_alignment_without_asking_the_llm_for_it() -> None:
    body = _valid_body()
    llm_response = "Picture 1 is the first frame.\n\n" + body

    normalized = normalize_h3_prompt_body(llm_response)
    final_prompt = compose_h3_prompt(llm_response, "i2v")

    assert normalized == body
    assert validate_h3_prompt_body(normalized) == (True, "")
    assert final_prompt == f"{I2V_ALIGNMENT}\n\n{body}"
    assert validate_h3_prompt(final_prompt, "i2v") == (True, "")


def test_i2v_prompt_messages_exclude_stored_generation_metadata() -> None:
    visual_context = (
        "visual_context:\nPicture 1: One anime character holds a book at chest "
        "height in a static centered composition."
    )
    messages = VideoMode._prompt_messages(
        "i2v",
        "머리카락과 옷이 약한 바람에 흔들린다",
        "[ANIMA_CONTENT] prior action [LORA_DATA] secret/path.safetensors",
        visual_context,
    )
    combined = "\n".join(str(message["content"]) for message in messages)

    assert "prior action" not in combined
    assert "secret/path.safetensors" not in combined
    assert I2V_ALIGNMENT not in combined
    assert "Picture 1 itself is the ultimate authority" in combined
    assert "sole authority for new motion and events" in combined
    assert visual_context in combined


def test_visual_context_stage_describes_static_visible_facts_only() -> None:
    messages = VideoMode._visual_context_messages("i2v")
    combined = "\n".join(str(message["content"]) for message in messages)

    assert "directly visible" in combined
    assert "Do not infer past or future actions" in combined
    assert "Describe a held object as being held" in combined
    assert "ANIMA_CONTENT" not in combined
    assert I2V_ALIGNMENT not in combined

    raw = "```text\nVisual context:\nPicture 1: A static centered portrait.\n```"
    assert normalize_visual_context(raw) == (
        "visual_context:\nPicture 1: A static centered portrait."
    )
    assert validate_visual_context(raw) == (True, "")


def _synthetic_i2v_api_workflow() -> dict:
    return {
        "122": {
            "class_type": "PrimitiveStringMultiline",
            "inputs": {"value": "old transport"},
            "_meta": {"title": "긍정프롬프트"},
        },
        "123": {
            "class_type": "RegexExtract",
            "inputs": {"string": ["122", 0]},
        },
        "124": {
            "class_type": "SoyaStringToFloat_mdsoya",
            "inputs": {"string": ["125", 0]},
        },
        "125": {
            "class_type": "RegexExtract",
            "inputs": {"string": ["122", 0]},
        },
        "126": {
            "class_type": "RegexExtract",
            "inputs": {"string": ["122", 0]},
        },
        "128": {
            "class_type": "SoyaStringToFloat_mdsoya",
            "inputs": {"string": ["127", 0]},
        },
        "127": {
            "class_type": "RegexExtract",
            "inputs": {"string": ["122", 0]},
        },
        "129": {
            "class_type": "SoyaFloatToInt_mdsoya",
            "inputs": {"float_value": ["124", 0]},
        },
        "130": {
            "class_type": "SoyaFloatToInt_mdsoya",
            "inputs": {"float_value": ["128", 0]},
        },
        "131": {
            "class_type": "LoadImagesFromPath_mdsoya",
            "inputs": {"path": ["123", 0]},
        },
        "133": {
            "class_type": "SoyaStringToFloat_mdsoya",
            "inputs": {"text": ["134", 0]},
        },
        "134": {
            "class_type": "RegexExtract",
            "inputs": {"string": ["122", 0]},
        },
        "135": {
            "class_type": "RegexExtract",
            "inputs": {"string": ["122", 0]},
        },
        "136": {
            "class_type": "SoyaStringToFloat_mdsoya",
            "inputs": {"text": ["135", 0]},
        },
        "137": {
            "class_type": "SoyaFloatToInt_mdsoya",
            "inputs": {"float_value": ["136", 0]},
        },
        "105:107": {
            "class_type": "ComfyMathExpression",
            "inputs": {"values.a": ["105:111", 0]},
        },
        "105:104": {
            "class_type": "MiniMaxH3ImageToVideo",
            "inputs": {
                "prompt": ["126", 0],
                "width": ["129", 0],
                "height": ["130", 0],
                "length": ["105:107", 1],
                "first_frame": ["131", 0],
            },
        },
        "105:111": {
            "class_type": "PrimitiveFloat",
            "inputs": {"value": ["133", 0]},
            "_meta": {"title": "Float (duration)"},
        },
        "105:15": {
            "class_type": "RandomNoise",
            "inputs": {"noise_seed": ["137", 0]},
        },
        "92": {
            "class_type": "SaveVideo",
            "inputs": {"filename_prefix": "video/SOYA_H3_I2V"},
        },
    }


def _synthetic_first_last_api_workflow() -> dict:
    workflow = _synthetic_i2v_api_workflow()
    workflow["135"] = {
        "class_type": "LoadImagesFromPath_mdsoya",
        "inputs": {"path": ["123", 0]},
    }
    workflow["138"] = {
        "class_type": "FilterImagesByName_mdsoya",
        "inputs": {
            "filter_names": "[1]",
            "mode": "include",
            "images": ["135", 0],
            "filenames": ["135", 1],
        },
    }
    workflow["139"] = {
        "class_type": "FilterImagesByName_mdsoya",
        "inputs": {
            "filter_names": "[2]",
            "mode": "include",
            "images": ["135", 0],
            "filenames": ["135", 1],
        },
    }
    workflow["105:104"]["inputs"]["first_frame"] = ["138", 0]
    workflow["105:104"]["inputs"]["last_frame"] = ["139", 0]
    return workflow


def test_real_h3_i2v_workflow_exposes_positive_transport_node() -> None:
    workflow_path = next(
        (ROOT / "comfy" / "user" / "default" / "workflows" / "SOYA_USER").glob(
            "*H3_I2V*첫프레임*.json"
        )
    )
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))

    positive_nodes = [
        node
        for node in workflow["nodes"]
        if node.get("type") == "PrimitiveStringMultiline"
        and node.get("title") == "긍정프롬프트"
    ]
    assert len(positive_nodes) == 1
    assert positive_nodes[0]["widgets_values"][0].startswith(
        "[PATH]\nsoya_video\n[PROMPT]\n"
    )


def test_real_h3_first_last_workflow_exposes_the_same_transport_contract() -> None:
    workflow_path = next(
        (ROOT / "comfy" / "user" / "default" / "workflows" / "SOYA_USER").glob(
            "*H3_I2V*첫마지막프레임*.json"
        )
    )
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    positive = next(
        node
        for node in workflow["nodes"]
        if node.get("type") == "PrimitiveStringMultiline"
        and node.get("title") == "긍정프롬프트"
    )["widgets_values"][0]

    assert positive.startswith("[PATH]\nsoya_video\n[PROMPT]\n")
    assert "\n[DURATION]\n5.0\n[SEED]\n123\n[END]" in positive


def test_i2v_transport_block_and_api_patch_drive_the_real_connected_inputs() -> None:
    prompt = _valid_body(I2V_ALIGNMENT)
    transport = build_i2v_workflow_block(prompt, 384, 672, 5.0, 123)

    assert transport == (
        "[PATH]\nsoya_video\n[PROMPT]\n"
        f"{prompt}\n[W]\n384\n[H]\n672\n"
        "[DURATION]\n5.0\n[SEED]\n123\n[END]"
    )

    patched = VideoMode._patch_i2v_api_workflow(
        _synthetic_i2v_api_workflow(),
        transport,
        "job-1",
    )

    assert patched["122"]["inputs"]["value"] == transport
    assert patched["105:111"]["inputs"]["value"] == ["133", 0]
    assert patched["105:15"]["inputs"]["noise_seed"] == ["137", 0]
    assert patched["92"]["inputs"]["filename_prefix"] == "video/soya_h3/job-1"


def test_i2v_api_patch_rejects_a_positive_block_not_connected_to_first_frame() -> None:
    workflow = _synthetic_i2v_api_workflow()
    workflow["131"]["inputs"]["path"] = ["unrelated", 0]

    with pytest.raises(RuntimeError, match="시작 이미지"):
        VideoMode._patch_i2v_api_workflow(
            workflow,
            build_i2v_workflow_block(
                _valid_body(I2V_ALIGNMENT),
                512,
                512,
                5.0,
                123,
            ),
            "job-2",
        )


def test_first_last_api_patch_requires_named_one_and_two_frame_filters() -> None:
    transport = build_i2v_workflow_block(
        _valid_body(FIRST_LAST_ALIGNMENT),
        512,
        512,
        5.0,
        456,
    )
    patched = VideoMode._patch_i2v_api_workflow(
        _synthetic_first_last_api_workflow(),
        transport,
        "first-last-job",
        "first_last",
    )

    assert patched["122"]["inputs"]["value"] == transport
    assert patched["105:104"]["inputs"]["first_frame"] == ["138", 0]
    assert patched["105:104"]["inputs"]["last_frame"] == ["139", 0]
    assert patched["138"]["inputs"]["filter_names"] == "[1]"
    assert patched["139"]["inputs"]["filter_names"] == "[2]"

    swapped = _synthetic_first_last_api_workflow()
    swapped["138"]["inputs"]["filter_names"] = "[2]"
    with pytest.raises(RuntimeError, match=r"\[1\]/\[2\]"):
        VideoMode._patch_i2v_api_workflow(
            swapped,
            transport,
            "invalid-first-last-job",
            "first_last",
        )


def test_overlay_is_scaled_after_high_resolution_render_and_applied_to_every_frame() -> None:
    high_res = Image.new("RGBA", (400, 400), (20, 20, 20, 255))
    overlay = Image.new("RGBA", (400, 480), (0, 0, 0, 0))
    mask = Image.new("L", (400, 480), 0)
    for x in range(100, 300):
        for y in range(40, 100):
            overlay.putpixel((x, y), (255, 255, 255, 255))
            mask.putpixel((x, y), 255)
    for x in range(400):
        for y in range(400, 480):
            overlay.putpixel((x, y), (0, 0, 0, 255))
            mask.putpixel((x, y), 255)
    frames = [
        Image.new("RGBA", (100, 100), (255, 0, 0, 255)),
        Image.new("RGBA", (100, 100), (0, 0, 255, 255)),
    ]

    composed = VideoMode._apply_overlay_to_frames(
        frames,
        high_res,
        overlay,
        mask,
    )

    assert [image.size for image in composed] == [(100, 120), (100, 120)]
    assert composed[0].getpixel((50, 15))[:3] == (255, 255, 255)
    assert composed[1].getpixel((50, 15))[:3] == (255, 255, 255)
    assert composed[0].getpixel((10, 110))[:3] == (0, 0, 0)


def test_animated_archive_writer_keeps_all_frames(tmp_path: Path) -> None:
    frames = [
        Image.new("RGBA", (32, 32), (255, 0, 0, 255)),
        Image.new("RGBA", (32, 32), (0, 0, 255, 255)),
    ]

    path, extension = VideoMode._save_animation(
        frames,
        str(tmp_path / "animation"),
        quality=80,
    )

    assert extension in (".avif", ".webp")
    with Image.open(path) as image:
        assert image.is_animated is True
        assert image.n_frames == 2


@pytest.mark.asyncio
async def test_i2v_build_uses_picture_only_and_program_adds_alignment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    raw_dir = tmp_path / "_raw"
    raw_dir.mkdir()
    Image.new("RGB", (512, 512), "green").save(
        raw_dir / "source.webp",
        format="WEBP",
    )
    (tmp_path / "source.json").write_text(
        json.dumps(
            {
                "positive": (
                    "[ANIMA_CONTENT] lowering book, speaking urgently "
                    "[LORA_DATA] secret/path.safetensors"
                )
            }
        ),
        encoding="utf-8",
    )
    body = _valid_body()
    calls = []
    visual_context = (
        "visual_context:\nPicture 1: One anime character is centered in a static "
        "upper-body composition, holding a book at chest height."
    )

    async def fake_vision_call(task_key, messages, **kwargs):
        calls.append("vision")
        combined = "\n".join(str(message["content"]) for message in messages)
        assert task_key == "video_prompt_i2v"
        assert "lowering book" not in combined
        assert "secret/path.safetensors" not in combined
        assert "머리카락과 옷이 약한 바람에 흔들린다" not in combined
        assert I2V_ALIGNMENT not in combined
        assert len(kwargs["images"]) == 1
        assert kwargs["result_validator"](visual_context) == (True, "")
        return visual_context

    async def fake_text_call(task_key, messages, **kwargs):
        calls.append("text")
        combined = "\n".join(str(message["content"]) for message in messages)
        assert task_key == "video_prompt_i2v"
        assert "lowering book" not in combined
        assert "secret/path.safetensors" not in combined
        assert "머리카락과 옷이 약한 바람에 흔들린다" in combined
        assert visual_context in combined
        assert "images" not in kwargs
        assert kwargs["result_validator"](body) == (True, "")
        return body

    monkeypatch.setattr(
        video_module.llm_service,
        "callLLMVisionTask",
        fake_vision_call,
    )
    monkeypatch.setattr(video_module.llm_service, "callLLMTask", fake_text_call)
    monkeypatch.setattr(video_module, "_log_lighbd_history", lambda _record: None)

    async def notify(_event_type, _data):
        return None

    mode = VideoMode()
    mode.get_backup_dir = lambda: str(tmp_path)
    mode.notify_frontend_func = notify

    def reject_stored_context(_name):
        raise AssertionError("I2V must not load the stored illustration prompt")

    monkeypatch.setattr(mode, "_source_context", reject_stored_context)

    result = await mode.build_prompt(
        {
            "mode": "i2v",
            "source_backup": "source",
            "instruction": "머리카락과 옷이 약한 바람에 흔들린다",
            "preset": "1:1",
        },
        queue_item_id="queue-i2v",
    )

    assert calls == ["vision", "text"]
    assert result["success"] is True
    assert result["h3_prompt"] == f"{I2V_ALIGNMENT}\n\n{body}"
    assert result["llm_trace"] == [
        "video_prompt:i2v:queue-i2v:visual_context",
        "video_prompt:i2v:queue-i2v",
    ]


@pytest.mark.asyncio
async def test_h3_llm_build_emits_live_events_and_full_history(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "source.json").write_text(
        json.dumps({"provider": "video", "positive": "a stored scene"}),
        encoding="utf-8",
    )
    notifications = []
    history = []

    async def notify(event_type, data):
        notifications.append((event_type, dict(data)))

    async def fake_call(task_key, messages, **kwargs):
        assert task_key == "video_prompt_t2v"
        assert "a stored scene" in messages[-1]["content"]
        kwargs["metadata_sink"].update(
            {"prompt_tokens": 123, "completion_tokens": 45, "ttft": 0.2}
        )
        await kwargs["stream_observer"]({"type": "delta", "text": "partial"})
        return _valid_body()

    monkeypatch.setattr(video_module.llm_service, "callLLMTask", fake_call)
    monkeypatch.setattr(
        video_module,
        "_log_lighbd_history",
        lambda record: history.append(dict(record)),
    )
    mode = VideoMode()
    mode.get_backup_dir = lambda: str(tmp_path)
    mode.notify_frontend_func = notify

    result = await mode.build_prompt(
        {
            "mode": "t2v",
            "source_backup": "source",
            "instruction": "The subject gently turns toward the camera.",
            "preset": "auto",
        },
        queue_item_id="queue-1",
    )

    assert result["success"] is True
    assert result["history_id"] == "video_prompt:t2v:queue-1"
    assert [data["type"] for event, data in notifications if event == "lighbd_llm_stream"] == [
        "start",
        "delta",
        "done",
    ]
    assert len(history) == 1
    assert history[0]["history_id"] == result["history_id"]
    assert "a stored scene" in history[0]["input"][-1]["content"]
    assert history[0]["output"] == _valid_body()
    assert history[0]["status"] == "ok"


@pytest.mark.asyncio
@pytest.mark.parametrize("render_mode", ["i2v", "first_last"])
async def test_render_archives_composite_and_raw_before_cleaning_mp4(
    tmp_path: Path,
    monkeypatch,
    render_mode: str,
) -> None:
    backup_dir = tmp_path / "backups"
    raw_dir = backup_dir / "_raw"
    comfy_input = tmp_path / "comfy_input"
    backup_dir.mkdir()
    raw_dir.mkdir()
    comfy_input.mkdir()
    Image.new("RGB", (800, 800), "green").save(
        raw_dir / "source.webp",
        format="WEBP",
    )
    (backup_dir / "source.json").write_text(
        json.dumps({"positive": "stored illustration prompt"}),
        encoding="utf-8",
    )
    if render_mode == "first_last":
        Image.new("RGB", (800, 800), "yellow").save(
            raw_dir / "last.webp",
            format="WEBP",
        )
        (backup_dir / "last.json").write_text(
            json.dumps({"positive": "stored last illustration prompt"}),
            encoding="utf-8",
        )
    workflow_path = tmp_path / "i2v.json"
    workflow_path.write_text(
        json.dumps(
            {
                "nodes": [
                    {
                        "id": 122,
                        "type": "PrimitiveStringMultiline",
                        "title": "긍정프롬프트",
                        "widgets_values": ["workflow default"],
                    }
                ],
                "links": [],
            }
        ),
        encoding="utf-8",
    )
    frames = [
        Image.new("RGBA", (512, 512), (255, 0, 0, 255)),
        Image.new("RGBA", (512, 512), (0, 0, 255, 255)),
    ]
    events = []
    submitted = {}

    async def convert(workflow, *, task_key):
        assert task_key == "video_generation"
        positive = next(node for node in workflow["nodes"] if node.get("id") == 122)
        assert positive["widgets_values"][0] == "workflow default"
        return (
            _synthetic_first_last_api_workflow()
            if render_mode == "first_last"
            else _synthetic_i2v_api_workflow()
        ), None

    async def submit(workflow, progress_callback=None, *, task_key):
        assert task_key == "video_generation"
        transport = workflow["122"]["inputs"]["value"]
        assert transport.startswith("[PATH]\nsoya_video\n[PROMPT]\n")
        expected_alignment = (
            FIRST_LAST_ALIGNMENT if render_mode == "first_last" else I2V_ALIGNMENT
        )
        assert _valid_body(expected_alignment) in transport
        assert "\n[W]\n512\n[H]\n512\n[DURATION]\n5.0\n[SEED]\n" in transport
        seed_text = transport.split("\n[SEED]\n", 1)[1].split("\n[END]", 1)[0]
        assert seed_text.isdigit()
        submitted["seed"] = int(seed_text)
        assert workflow["105:111"]["inputs"]["value"] == ["133", 0]
        assert workflow["105:15"]["inputs"]["noise_seed"] == ["137", 0]
        staged = comfy_input / "soya_video" / "[1].png"
        assert staged.is_file()
        with Image.open(staged) as image:
            assert image.size == (512, 512)
        last_staged = comfy_input / "soya_video" / "[2].png"
        if render_mode == "first_last":
            assert last_staged.is_file()
            with Image.open(last_staged) as image:
                assert image.size == (512, 512)
                red, green, blue = image.convert("RGB").getpixel((0, 0))
                assert red >= 250 and green >= 250 and blue <= 5
        else:
            assert not last_staged.exists()
        return b"temporary-mp4", {
            "filename": "temporary.mp4",
            "subfolder": "video/soya_h3",
            "type": "output",
        }

    async def cleanup(descriptor, *, task_key):
        assert task_key == "video_generation"
        assert descriptor["filename"] == "temporary.mp4"
        # Cleanup must happen only after both animated files and metadata exist.
        assert len(list(backup_dir.glob("*.avif"))) + len(list(backup_dir.glob("*.webp"))) == 1
        archived_raw = [
            path
            for pattern in ("*.avif", "*.webp")
            for path in raw_dir.glob(pattern)
            if path.stem not in {"source", "last"}
        ]
        assert len(archived_raw) == 1
        assert len(list(backup_dir.glob("*_info.json"))) == 1
        events.append("mp4_cleaned")
        return True

    async def notify(event_type, data):
        events.append((event_type, data.get("name")))

    monkeypatch.setattr(
        VideoMode,
        "_decode_mp4_frames",
        staticmethod(lambda _mp4: [frame.copy() for frame in frames]),
    )
    mode = VideoMode()
    mode.get_backup_dir = lambda: str(backup_dir)
    mode.get_config = lambda: {
        "comfy_input_dir": str(comfy_input),
        "video_workflow_source_paths": {render_mode: str(workflow_path)},
        "backup_webp_quality": 80,
    }
    mode.convert_workflow_func = convert
    mode.submit_workflow_func = submit
    mode.cleanup_comfy_video_func = cleanup
    mode.cleanup_backups_func = lambda: events.append("retention_checked")
    mode.invalidate_backup_cache_func = lambda: events.append("cache_invalidated")
    mode.notify_frontend_func = notify

    render_params = {
        "mode": render_mode,
        "source_backup": "source",
        "instruction": "move gently",
        "preset": "1:1",
        "h3_prompt": _valid_body(
            FIRST_LAST_ALIGNMENT if render_mode == "first_last" else I2V_ALIGNMENT
        ),
        "llm_trace": ["trace-1"],
    }
    if render_mode == "first_last":
        render_params["last_backup"] = "last"
    result = await mode.render_video(
        render_params,
        queue_item_id="gpu-1",
    )

    assert result["success"] is True
    assert result["preset"] == "1:1"
    assert not (comfy_input / "soya_video").exists()
    assert events[-1] == "mp4_cleaned"
    info = json.loads(
        next(backup_dir.glob("*_info.json")).read_text(encoding="utf-8")
    )
    assert info["source_backup"] == "source"
    assert info["llm_trace"] == ["trace-1"]
    assert info["video_duration_seconds"] == 5.0
    assert info["video_seed"] == submitted["seed"]
