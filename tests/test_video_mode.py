from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

import modes.video_mode as video_module
import modes.video_postprocess as postprocess_module
from modes.video_mode import (
    FAST_PRESETS,
    FIRST_LAST_ALIGNMENT,
    I2V_ALIGNMENT,
    VideoMode,
    alignment_for_mode,
    build_i2v_workflow_block,
    center_crop_to_ratio,
    choose_fast_preset,
    compose_h3_prompt,
    normalize_h3_prompt_body,
    normalize_video_duration,
    normalize_visual_context,
    parse_auto_visual_direction,
    validate_h3_prompt,
    validate_h3_prompt_body,
    validate_auto_visual_direction,
    validate_visual_context,
)


ROOT = Path(__file__).resolve().parents[1]


def test_video_postprocess_accepts_all_three_upscale_models() -> None:
    for model in ("realesr-animevideov3", "anime4k-fast-m", "lanczos"):
        normalized = postprocess_module.normalize_video_postprocess_config(
            {"enabled": True, "scale": 2, "model": model}
        )
        assert normalized["model"] == model
    with pytest.raises(ValueError, match="지원하지 않는"):
        postprocess_module.normalize_video_postprocess_config(
            {"enabled": True, "scale": 2, "model": "unknown"}
        )


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
    assert validate_h3_prompt(_valid_body(I2V_ALIGNMENT), "i2v") == (True, "")
    assert validate_h3_prompt(_valid_body(FIRST_LAST_ALIGNMENT), "first_last") == (
        True,
        "",
    )
    assert validate_h3_prompt(_valid_body(), "i2v")[0] is False


def test_program_adds_exact_i2v_alignment_without_asking_the_llm_for_it() -> None:
    body = _valid_body()
    llm_response = "Picture 1 is the first frame.\n\n" + body

    normalized = normalize_h3_prompt_body(llm_response)
    final_prompt = compose_h3_prompt(llm_response, "i2v")

    assert normalized == body
    assert validate_h3_prompt_body(normalized) == (True, "")
    assert final_prompt == f"{I2V_ALIGNMENT}\n\n{body}"
    assert validate_h3_prompt(final_prompt, "i2v") == (True, "")


def test_duration_supports_every_whole_second_from_one_through_fifteen() -> None:
    for duration in range(1, 16):
        assert normalize_video_duration(duration) == float(duration)
    for invalid in (0, 16, 1.5, True, "not-a-number"):
        with pytest.raises(ValueError, match="1초부터 15초"):
            normalize_video_duration(invalid)


def test_dynamic_duration_updates_flf_alignment_and_h3_prompt_context() -> None:
    duration = 12
    alignment = alignment_for_mode("first_last", duration)
    prompt = compose_h3_prompt(_valid_body(), "first_last", duration)
    messages = VideoMode._prompt_messages(
        "first_last",
        "move gently",
        "",
        "visual_context:\nPicture 1: start. Picture 2: end.",
        duration=duration,
    )
    combined = "\n".join(str(message["content"]) for message in messages)

    assert "12.00-second mark" in alignment
    assert prompt.startswith(alignment)
    assert validate_h3_prompt(prompt, "first_last", duration) == (True, "")
    assert "final 12-second H3 prompt" in combined
    assert "by 12.00 seconds" in combined


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


def test_auto_visual_direction_prefers_two_string_json_array() -> None:
    raw = json.dumps(
        [
            "Picture 1: A centered character holds a closed book at chest height.",
            "The character slowly looks up, loosens their grip, and takes one measured step forward while the camera gently pushes in.",
        ]
    )

    visual_context, direction = parse_auto_visual_direction(raw)

    assert visual_context == (
        "visual_context:\n"
        "Picture 1: A centered character holds a closed book at chest height."
    )
    assert direction.startswith("The character slowly looks up")
    assert validate_auto_visual_direction(raw) == (True, "")


def test_auto_visual_direction_joins_adjacent_single_item_arrays(
    capsys: pytest.CaptureFixture[str],
) -> None:
    context = (
        "Picture 1: A shy young woman holds a red-bound book in a sunlit library."
    )
    direction = (
        "The young woman glances down at the book, steadies her breath, then "
        "looks up and delivers her line with a determined nod."
    )
    raw = f"{json.dumps([context])}\n\n{json.dumps([direction])}"

    visual_context, parsed_direction = parse_auto_visual_direction(raw)

    assert visual_context == f"visual_context:\n{context}"
    assert parsed_direction == direction
    assert "imagine and describe one coherent" not in parsed_direction
    assert validate_auto_visual_direction(raw) == (True, "")
    assert "adjacent_json_arrays_joined" in capsys.readouterr().out


def test_auto_visual_direction_repairs_usable_response_locally() -> None:
    context = "Picture 1: A character stands beside a window."
    object_response = json.dumps(
        {"misspelled_context_key": context, "anything": "They turn toward the light."}
    )
    visual_context, direction = parse_auto_visual_direction(object_response)
    assert visual_context.endswith(context)
    assert direction == "They turn toward the light."

    fenced_trailing_comma = (
        "```json\n"
        f'["{context}", "They blink.", "The curtain settles.",]\n'
        "```"
    )
    visual_context, direction = parse_auto_visual_direction(fenced_trailing_comma)
    assert visual_context.endswith(context)
    assert direction == "They blink.\n\nThe curtain settles."

    visual_context, direction = parse_auto_visual_direction(
        json.dumps([context, ""])
    )
    assert visual_context.endswith(context)
    assert "imagine and describe one coherent" in direction


def test_auto_visual_direction_retries_when_visual_context_is_missing() -> None:
    for invalid in (
        '["", "They turn toward the light."]',
        '["", ""]',
        "[]",
        "unstructured response without visual context",
    ):
        assert validate_auto_visual_direction(invalid)[0] is False


def test_auto_visual_direction_prompt_receives_duration_and_mode_contract() -> None:
    messages = VideoMode._visual_context_messages(
        "first_last",
        auto_instruction=True,
        duration=12,
        dialogue_contexts=[
            ("Picture 1", 'Alice: "기다렸어." #relieved'),
            ("Picture 2", "Alice: (이제 안심해도 되겠지.) #hopeful"),
        ],
    )
    combined = "\n".join(str(message["content"]) for message in messages)

    assert "exactly two strings" in combined
    assert "12 seconds" in combined
    assert "12.00 seconds" in combined
    assert "MiniMax H3" in combined
    assert "user-authored video direction is available" in combined
    assert 'Alice: "기다렸어." #relieved' in combined
    assert "Alice: (이제 안심해도 되겠지.) #hopeful" in combined
    assert "Picture 1 backup dialogue and emotion context" in combined
    assert "Picture 2 backup dialogue and emotion context" in combined
    assert "must never enter the first string" in combined
    assert "meaningfully consistent" in combined
    assert "Preserve supplied dialogue verbatim" in combined
    assert "parenthesized thoughts remain internal" in combined


def test_static_visual_context_stage_does_not_receive_backup_dialogue() -> None:
    messages = VideoMode._visual_context_messages(
        "i2v",
        dialogue_contexts=[("Picture 1", 'Alice: "숨겨진 대사" #angry')],
    )
    combined = "\n".join(str(message["content"]) for message in messages)

    assert "숨겨진 대사" not in combined


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
    workflow_path = (
        ROOT
        / "comfy"
        / "user"
        / "default"
        / "workflows"
        / "SOYA_USER"
        / "배포_영상_H3_I2V_v1.json"
    )
    assert workflow_path.is_file()
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
    workflow_path = (
        ROOT
        / "comfy"
        / "user"
        / "default"
        / "workflows"
        / "SOYA_USER"
        / "배포_영상_H3_FLF2V_v1.json"
    )
    assert workflow_path.is_file()
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
async def test_i2v_auto_instruction_is_generated_in_visual_call_once(
    tmp_path: Path,
    monkeypatch,
) -> None:
    raw_dir = tmp_path / "_raw"
    raw_dir.mkdir()
    Image.new("RGB", (512, 512), "green").save(
        raw_dir / "source.webp",
        format="WEBP",
    )
    dialogue_context = 'Alice: "괜찮아, 이제 말해도 돼." #reassuring'
    (tmp_path / "source_info.json").write_text(
        json.dumps({"speak_text": dialogue_context}, ensure_ascii=False),
        encoding="utf-8",
    )
    visual_value = "Picture 1: One character stands centered in a quiet room."
    direction_value = (
        'Alice gives a reassuring look and quietly says "괜찮아, 이제 말해도 돼." '
        "while making a small, inviting hand gesture."
    )
    visual_response = json.dumps([visual_value, direction_value])
    body = _valid_body()
    calls = []

    async def fake_vision_call(task_key, messages, **kwargs):
        calls.append("vision")
        combined = "\n".join(str(message["content"]) for message in messages)
        assert task_key == "video_prompt_i2v"
        assert "5-second video" in combined
        assert dialogue_context in combined
        assert "semantic authority for the depicted dramatic moment" in combined
        assert "#emotion annotations as performance guidance" in combined
        assert kwargs["result_validator"](visual_response) == (True, "")
        return visual_response

    async def fake_text_call(task_key, messages, **kwargs):
        calls.append("text")
        combined = "\n".join(str(message["content"]) for message in messages)
        assert task_key == "video_prompt_i2v"
        assert direction_value in combined
        assert f"visual_context:\n{visual_value}" in combined
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

    result = await mode.build_prompt(
        {
            "mode": "i2v",
            "source_backup": "source",
            "auto_instruction": True,
            "instruction": "이 값은 자동 모드에서 사용되면 안 된다",
            "preset": "1:1",
            "duration": 5,
        },
        queue_item_id="queue-auto-i2v",
    )

    assert calls == ["vision", "text"]
    assert result["success"] is True
    assert result["instruction"] == direction_value
    assert result["llm_trace"] == [
        "video_prompt:i2v:queue-auto-i2v:visual_context_auto_direction",
        "video_prompt:i2v:queue-auto-i2v",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("render_mode", ["i2v", "first_last"])
async def test_render_spools_video_postprocess_before_cleaning_comfy_mp4(
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
        expected_alignment = alignment_for_mode(render_mode, 12)
        assert _valid_body(expected_alignment) in transport
        assert "\n[W]\n512\n[H]\n512\n[DURATION]\n12.0\n[SEED]\n" in transport
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
        # Comfy output cleanup is allowed only after the independent queue spool
        # has durable MP4 bytes and a recovery manifest.
        jobs = list((backup_dir / "_video_postprocess_spool").iterdir())
        assert len(jobs) == 1
        assert (jobs[0] / "input.mp4").read_bytes() == b"temporary-mp4"
        assert (jobs[0] / "job.json").is_file()
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
        "duration": 12,
        "h3_prompt": _valid_body(alignment_for_mode(render_mode, 12)),
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
    job_dir = Path(result["postprocess_job"]["job_dir"])
    manifest = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
    assert manifest["source_backup"] == "source"
    assert manifest["llm_trace"] == ["trace-1"]
    assert manifest["duration"] == 12.0
    assert manifest["video_seed"] == submitted["seed"]
    assert manifest["upscale_enabled"] is True
    assert manifest["upscale_scale"] == 2
    assert manifest["output_width"] == 1024
    assert not list(backup_dir.glob("*.avif"))


@pytest.mark.asyncio
async def test_video_postprocess_commits_verified_pair_and_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    backup_dir = tmp_path / "backups"
    job_dir = backup_dir / "_video_postprocess_spool" / "i2v-job"
    job_dir.mkdir(parents=True)
    (job_dir / "input.mp4").write_bytes(b"staged-mp4")
    manifest = {
        "version": 1,
        "spool_id": "i2v-job",
        "base_name": "20260812_120000_deadbeef",
        "mode": "i2v",
        "source_backup": "source",
        "last_backup": "",
        "positive": "synthetic H3 prompt",
        "instruction": "move gently",
        "llm_trace": ["trace-1"],
        "preset": "1:1",
        "source_width": 512,
        "source_height": 512,
        "output_width": 1024,
        "output_height": 1024,
        "raw_output_height": 1024,
        "duration": 5.0,
        "fps": 24,
        "video_seed": 123,
        "render_elapsed": 2.5,
        "quality": 80,
        "upscale_enabled": True,
        "upscale_scale": 2,
        "upscale_model": "realesr-animevideov3",
        "source_info": {"bot_name": "test-bot"},
    }
    (job_dir / "job.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    async def fake_process(path, *, settings, progress_callback=None):
        assert Path(path) == job_dir.resolve()
        assert settings["scale"] == 2
        main = job_dir / "result_main.avif"
        raw = job_dir / "result_raw.avif"
        main.write_bytes(b"verified-main")
        raw.write_bytes(b"verified-raw")
        if progress_callback:
            await progress_callback({"phase": "video_postprocess_validated", "percentage": 97})
        return {
            "manifest": manifest,
            "main_path": str(main),
            "raw_path": str(raw),
            "extension": ".avif",
            "frame_count": 120,
            "upscale_enabled": True,
            "upscale_scale": 2,
        }

    events = []
    monkeypatch.setattr(video_module, "process_staged_video", fake_process)
    mode = VideoMode()
    mode.get_backup_dir = lambda: str(backup_dir)
    mode.get_config = lambda: {
        "video_postprocess": {
            "enabled": True,
            "scale": 2,
            "model": "realesr-animevideov3",
            "gpu_id": "auto",
            "tile_size": 0,
            "worker_count": 1,
        }
    }
    mode.cleanup_backups_func = lambda: events.append("retention")
    mode.invalidate_backup_cache_func = lambda: events.append("invalidate")

    async def notify(event_type, data):
        events.append((event_type, data))

    mode.notify_frontend_func = notify
    result = await mode.postprocess_staged_video(
        {"job_dir": str(job_dir)},
        queue_item_id="post-1",
    )

    base_name = manifest["base_name"]
    assert result["backup_name"] == base_name
    assert (backup_dir / f"{base_name}.avif").read_bytes() == b"verified-main"
    assert (backup_dir / "_raw" / f"{base_name}.avif").read_bytes() == b"verified-raw"
    info = json.loads(
        (backup_dir / f"{base_name}_info.json").read_text(encoding="utf-8")
    )
    assert info["video_source_width"] == 512
    assert info["video_width"] == 1024
    assert info["video_upscale_scale"] == 2
    assert info["bot_name"] == "test-bot"
    assert not job_dir.exists()
    assert events[-1] == ("backup_created", {"name": base_name})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "expected_runner"),
    [
        ("realesr-animevideov3", "realesrgan"),
        ("anime4k-fast-m", "anime4k"),
        ("lanczos", "lanczos"),
        ("", "none"),
    ],
)
async def test_staged_postprocess_routes_upscaler_and_honors_webp_choice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    expected_runner: str,
) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "input.mp4").write_bytes(b"mp4")
    manifest = {
        "duration": 1,
        "fps": 2,
        "quality": 80,
        "output_height": 16,
        "upscale_enabled": bool(model),
        "upscale_scale": 2,
        "upscale_model": model,
        "output_format": "webp",
    }
    (job_dir / "job.json").write_text(json.dumps(manifest), encoding="utf-8")
    runners: list[str] = []

    async def fake_ensure_ffmpeg() -> Path:
        return tmp_path / "ffmpeg.exe"

    async def fake_command(command, *, label, cwd=None):
        assert label == "DECODE"
        output_dir = Path(command[-1]).parent
        for index in range(2):
            Image.new("RGB", (8, 8), "blue").save(
                output_dir / f"frame_{index:08d}.png"
            )
        return ""

    async def fake_runner(input_dir, output_dir, **kwargs):
        runners.append(expected_runner)
        output_dir.mkdir()
        for source in sorted(input_dir.glob("*.png")):
            Image.open(source).resize((16, 16)).save(output_dir / source.name)

    async def fake_encode_pair(frames_dir, directory, **kwargs):
        assert kwargs["output_format"] == "webp"
        assert len(list(frames_dir.glob("*.png"))) == 2
        main = directory / "result_main.webp"
        raw = directory / "result_raw.webp"
        main.write_bytes(b"main")
        raw.write_bytes(b"raw")
        return main, raw, ".webp"

    monkeypatch.setattr(postprocess_module, "ensure_ffmpeg", fake_ensure_ffmpeg)
    monkeypatch.setattr(postprocess_module, "_run_command", fake_command)
    monkeypatch.setattr(postprocess_module, "_encode_pair", fake_encode_pair)
    monkeypatch.setattr(postprocess_module, "_run_realesrgan", fake_runner)
    monkeypatch.setattr(postprocess_module, "_run_anime4k", fake_runner)
    monkeypatch.setattr(postprocess_module, "_run_lanczos", fake_runner)

    result = await postprocess_module.process_staged_video(
        job_dir,
        settings={"enabled": True, "scale": 2, "model": "realesr-animevideov3"},
    )

    assert runners == ([] if expected_runner == "none" else [expected_runner])
    assert result["extension"] == ".webp"
    assert result["output_format_requested"] == "webp"
    assert result["upscale_model"] == model
