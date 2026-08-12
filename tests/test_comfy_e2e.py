from __future__ import annotations

import io
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest
from PIL import Image

import comfy_installer.e2e as e2e_module
from comfy_installer.e2e import (
    ComfyE2EError,
    ComfyProcess,
    WorkflowValidation,
    _build_prompt_request,
    _validate_prompt_structure,
    bypass_sageattention_nodes,
    make_e2e_prompt,
    prepare_e2e_fixtures,
    promote_generated_fixture,
    protected_e2e_fixtures,
)


def test_compatibility_e2e_bypasses_sageattention_without_mutating_prompt() -> None:
    prompt = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {}},
        "2": {
            "class_type": "PathchSageAttentionKJ",
            "inputs": {"model": ["1", 0], "sage_attention": "auto"},
        },
        "3": {
            "class_type": "KSampler",
            "inputs": {"model": ["2", 0]},
        },
    }

    compatible, bypassed = bypass_sageattention_nodes(
        prompt,
        filename="fixture.json",
    )

    assert "2" not in compatible
    assert compatible["3"]["inputs"]["model"] == ["1", 0]
    assert bypassed == [
        {"node_id": "2", "class_type": "PathchSageAttentionKJ"}
    ]
    assert prompt["3"]["inputs"]["model"] == ["2", 0]


def test_compatibility_e2e_bypasses_chained_sageattention_nodes() -> None:
    prompt = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {}},
        "2": {
            "class_type": "PathchSageAttentionKJ",
            "inputs": {"model": ["1", 0]},
        },
        "3": {
            "class_type": "PathchSageAttentionKJ",
            "inputs": {"model": ["2", 0]},
        },
        "4": {"class_type": "KSampler", "inputs": {"model": ["3", 0]}},
    }

    compatible, bypassed = bypass_sageattention_nodes(
        prompt,
        filename="chained.json",
    )

    assert compatible["4"]["inputs"]["model"] == ["1", 0]
    assert {item["node_id"] for item in bypassed} == {"2", "3"}


def test_comfy_output_aggregates_repeated_lora_warnings(
    tmp_path: Path,
) -> None:
    messages: list[str] = []
    process = ComfyProcess(
        comfy_root=tmp_path,
        python=tmp_path / "python.exe",
        cancel_event=Event(),
        log=messages.append,
        port=12345,
    )

    for index in range(205):
        process._emit_output_line(
            f"lora key not loaded: fixture_{index}.weight"
        )
    process._emit_output_summary()

    assert len(messages) == 13
    assert messages[0].startswith("[Comfy][WARNING] lora key not loaded:")
    assert "fixture_9.weight" in messages[9]
    assert "누적 100건" in messages[10]
    assert "누적 200건" in messages[11]
    assert "최종 합계 205건" in messages[12]


def test_comfy_output_detects_prompt_worker_fatal_exit(tmp_path: Path) -> None:
    process = ComfyProcess(
        comfy_root=tmp_path,
        python=tmp_path / "python.exe",
        cancel_event=Event(),
        port=12345,
    )

    process._emit_output_line("Exception in thread Thread-10 (prompt_worker):")
    process._emit_output_line("torch.AcceleratorError: CUDA error: out of memory")

    detail = process.fatal_error()
    assert detail is not None
    assert "prompt_worker" in detail
    assert "out of memory" in detail


def test_comfy_process_keeps_explicit_extra_launch_args(tmp_path: Path) -> None:
    process = ComfyProcess(
        comfy_root=tmp_path,
        python=tmp_path / "python.exe",
        cancel_event=Event(),
        port=12345,
        extra_args=("--vram-headroom", "2"),
    )

    assert process.extra_args == ("--vram-headroom", "2")


def test_make_e2e_prompt_disables_private_assets_and_minimizes_work() -> None:
    structured = (
        "portrait\n"
        "[FACE_ID_DIR]\nprivate-face-directory\n"
        "[STYLE_DIR]\nprivate-style-directory\n"
        "[LORA_ACTIVATE]\ntrue\n"
        "[LORA_DATA]\n{\"list\":[{\"lora_path\":\"private.safetensors\"}]}\n"
        "[FACE_LORA_ACTIVATE]\ntrue\n"
        "[HRF_ACTIVATE]\ntrue\n"
        "[IMG_W]\n1024\n"
        "[IMG_H]\n1024\n"
        "[STEPS]\n30\n"
        "[END]\n"
    )
    validation = WorkflowValidation(
        binding_keys=("asset_workflow_source_path",),
        filename="asset.json",
        node_count=3,
        class_count=3,
        classes=(
            "EmptyLatentImage",
            "KSampler",
            "PrimitiveStringMultiline",
        ),
        prompt={
            "1": {
                "class_type": "PrimitiveStringMultiline",
                "inputs": {"value": structured},
            },
            "2": {
                "class_type": "KSampler",
                "inputs": {"steps": 32, "cfg": 5.0},
            },
            "3": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "width": 1024,
                    "height": 1024,
                    "batch_size": 4,
                },
            },
        },
        workflow={"nodes": []},
    )

    prompt = make_e2e_prompt(
        validation,
        face_input_relative="comfy-installer-e2e/face",
    )
    value = prompt["1"]["inputs"]["value"]
    assert "[FACE_ID_DIR]\ncomfy-installer-e2e/face\n" in value
    assert "[STYLE_DIR]\ncomfy-installer-e2e/face\n" in value
    assert "[LORA_ACTIVATE]\nfalse\n" in value
    assert "[LORA_DATA]\n{\"list\":[]}\n" in value
    assert "[FACE_LORA_ACTIVATE]\nfalse\n" in value
    assert "[HRF_ACTIVATE]\nfalse\n" in value
    assert "[IMG_W]\n512\n" in value
    assert "[IMG_H]\n512\n" in value
    assert "[STEPS]\n1\n" in value
    assert "private.safetensors" not in value
    assert prompt["2"]["inputs"]["steps"] == 1
    assert prompt["3"]["inputs"] == {
        "width": 512,
        "height": 512,
        "batch_size": 1,
    }
    assert validation.prompt["2"]["inputs"]["steps"] == 32


def test_make_e2e_prompt_accepts_h3_runtime_dimensions_and_steps() -> None:
    validation = WorkflowValidation(
        binding_keys=("video_workflow_source_paths.t2v",),
        filename="h3.json",
        node_count=1,
        class_count=1,
        classes=("MiniMaxH3TextToVideo",),
        prompt={
            "1": {
                "class_type": "MiniMaxH3TextToVideo",
                "inputs": {
                    "steps": 20,
                    "width": 1344,
                    "height": 768,
                },
            }
        },
        workflow={"nodes": []},
    )

    prompt = make_e2e_prompt(
        validation,
        sample_steps=8,
        sample_width=960,
        sample_height=544,
    )

    assert prompt["1"]["inputs"] == {
        "steps": 8,
        "width": 960,
        "height": 544,
    }


def test_make_e2e_prompt_keeps_illustration_cache_fields_as_json() -> None:
    structured = (
        "portrait\n"
        "[FACE_ID_DIR]\n"
        '{"list":[{"ipa_path":"private.ipadpt"}]}\n'
        "[CACHE_PATH]\n"
        '{"list":[{"emb_path":"private.pt"}]}\n'
        "[END]\n"
    )
    validation = WorkflowValidation(
        binding_keys=("illustration_workflow_source_paths.v3_anima",),
        filename="illustration.json",
        node_count=1,
        class_count=1,
        classes=("PrimitiveStringMultiline",),
        prompt={
            "1": {
                "class_type": "PrimitiveStringMultiline",
                "inputs": {"value": structured},
            }
        },
        workflow={"nodes": []},
    )

    prompt = make_e2e_prompt(
        validation,
        face_input_relative="comfy-installer-e2e/face",
    )
    value = prompt["1"]["inputs"]["value"]
    assert "[FACE_ID_DIR]\n{\"list\":[]}\n" in value
    assert "[CACHE_PATH]\n{\"list\":[]}\n" in value
    assert "comfy-installer-e2e/face" not in value
    assert "private.ipadpt" not in value
    assert "private.pt" not in value


def test_make_e2e_prompt_sets_training_and_edit_fixture_paths() -> None:
    training = WorkflowValidation(
        binding_keys=("lora_training_workflow_source_paths.anima",),
        filename="training.json",
        node_count=2,
        class_count=1,
        classes=("PrimitiveStringMultiline",),
        prompt={
            "1": {
                "class_type": "PrimitiveStringMultiline",
                "inputs": {
                    "value": (
                        "tag\n"
                        "[PROFILE]\nanima\n"
                        "[N_IMG]\n5\n"
                        "[STEP_PER_IMAGE]\n125\n"
                        "[SAVE_PER_STEP]\n25\n"
                        "[MULTI_IMG_FOLDER_NAME]\nprivate\n"
                        "[LORA_SAVE_PATH]\nprivate-output\n"
                        "[GEN_W]\n1024\n"
                        "[GEN_H]\n1024\n"
                        "[UPSCALE]\ntrue\n"
                        "[RESOLUTION]\n1024\n"
                        "[SAVE_AFTER]\n0\n"
                        "[TEST_POSITIVE]\nprivate\n"
                        "[TEST_NEGATIVE]\nprivate\n"
                        "[DIM]\n32\n"
                        "[ALPHA]\n16\n"
                        "[END]\n"
                    )
                },
            },
            "2": {
                "class_type": "PrimitiveStringMultiline",
                "inputs": {
                    "value": (
                        "[1]private negative one\n"
                        "[2]private negative two"
                    )
                },
            },
        },
        workflow={"nodes": []},
    )
    prompt = make_e2e_prompt(training)
    value = prompt["1"]["inputs"]["value"]
    assert "[MULTI_IMG_FOLDER_NAME]\ncomfy-installer-e2e/training\n" in value
    assert value.startswith("[1]1girl, portrait\n[PROFILE]\n")
    assert "[STEP_PER_IMAGE]\n1\n" in value
    assert "[RESOLUTION]\n1024\n" in value
    assert "[DIM]\n4\n" in value
    assert "private-output" not in value
    assert prompt["2"]["inputs"]["value"] == "[1]low quality"

    edit = WorkflowValidation(
        binding_keys=("qwen_edit_workflow_source_path",),
        filename="edit.json",
        node_count=1,
        class_count=1,
        classes=("PrimitiveStringMultiline",),
        prompt={
            "1": {
                "class_type": "PrimitiveStringMultiline",
                "inputs": {
                    "value": (
                        "[EDIT_PROMPT]\ntest\n"
                        "[IMAGE_PATH]\nold\n"
                        "[MASK_PATH]\nold\n"
                        "[STEPS]\n6\n"
                        "[FILENAME_PREFIX]\nold\n"
                        "[END]\n"
                    )
                },
            }
        },
        workflow={"nodes": []},
    )
    value = make_e2e_prompt(edit)["1"]["inputs"]["value"]
    assert value.count("comfy-installer-e2e/edit") == 2
    assert "[STEPS]\n1\n" in value
    assert "old" not in value


def test_make_e2e_prompt_replaces_entire_embedding_filter_value() -> None:
    validation = WorkflowValidation(
        binding_keys=("face_extract_workflow_source_path",),
        filename="face-extract.json",
        node_count=1,
        class_count=1,
        classes=("PrimitiveStringMultiline",),
        prompt={
            "1": {
                "class_type": "PrimitiveStringMultiline",
                "inputs": {
                    "value": (
                        "[PATH]\nprivate\n"
                        "[EMB_TARGET]\n[1]\n"
                        "[END]\n"
                    )
                },
            }
        },
        workflow={"nodes": []},
    )

    value = make_e2e_prompt(validation)["1"]["inputs"]["value"]
    assert "[EMB_TARGET]\nrepresentation\n[END]\n" in value
    assert "[1]" not in value


def test_make_e2e_prompt_uses_immutable_face_fixture_for_face_tags() -> None:
    validation = WorkflowValidation(
        binding_keys=("tag_analysis_workflow_source_path",),
        filename="face-tags.json",
        node_count=1,
        class_count=1,
        classes=("SoyaRefImageLoader_mdsoya",),
        prompt={
            "1": {
                "class_type": "SoyaRefImageLoader_mdsoya",
                "inputs": {
                    "image": "private.webp",
                    "fallback_width": 1024,
                    "fallback_height": 1024,
                },
            }
        },
        workflow={"nodes": []},
    )

    prompt = make_e2e_prompt(validation)

    assert prompt["1"]["inputs"]["image"] == (
        "comfy-installer-e2e-face.png"
    )


def test_make_e2e_prompt_selects_sdxl_profile_for_sdxl_training() -> None:
    validation = WorkflowValidation(
        binding_keys=("lora_training_workflow_source_paths.sdxl",),
        filename="sdxl-training.json",
        node_count=1,
        class_count=1,
        classes=("PrimitiveStringMultiline",),
        prompt={
            "1": {
                "class_type": "PrimitiveStringMultiline",
                "inputs": {
                    "value": (
                        "[1]private prompt\n"
                        "[PROFILE]\nanima\n"
                        "[N_IMG]\n5\n"
                        "[MULTI_IMG_FOLDER_NAME]\nprivate\n"
                        "[LORA_SAVE_PATH]\nprivate-output\n"
                        "[END]\n"
                    )
                },
            }
        },
        workflow={"nodes": []},
    )

    value = make_e2e_prompt(validation)["1"]["inputs"]["value"]
    assert value.startswith("[1]1girl, portrait\n[PROFILE]\nsdxl\n")
    assert "[PROFILE]\nanima\n" not in value


def test_prompt_request_includes_original_ui_workflow_for_rgthree_seed() -> None:
    prompt = {
        "290": {
            "class_type": "Seed (rgthree)",
            "inputs": {"seed": -1},
        }
    }
    workflow = {
        "nodes": [
            {
                "id": 290,
                "type": "Seed (rgthree)",
                "widgets_values": [-1],
            }
        ]
    }
    request = _build_prompt_request(
        prompt=prompt,
        workflow=workflow,
        filename="illustration.json",
        client_id="test-client",
    )

    pnginfo = request["extra_data"]["extra_pnginfo"]
    assert pnginfo["workflow"]["nodes"][0]["id"] == 290
    assert pnginfo["comfy_installer_e2e"] == "illustration.json"
    assert request["client_id"] == "test-client"
    request["prompt"]["290"]["inputs"]["seed"] = 123
    request["extra_data"]["extra_pnginfo"]["workflow"]["nodes"][0][
        "widgets_values"
    ][0] = 123
    assert prompt["290"]["inputs"]["seed"] == -1
    assert workflow["nodes"][0]["widgets_values"] == [-1]


def test_validate_prompt_structure_finds_missing_class_and_link() -> None:
    with pytest.raises(ComfyE2EError, match="로드되지 않은 class_type"):
        _validate_prompt_structure(
            prompt={
                "1": {
                    "class_type": "MissingNode",
                    "inputs": {"image": ["99", 0]},
                }
            },
            object_info={},
            filename="bad.json",
        )


def test_validate_prompt_structure_defers_required_inputs_to_comfy_runtime() -> None:
    classes = _validate_prompt_structure(
        prompt={
            "1": {
                "class_type": "DynamicNode",
                "inputs": {"expression": "a + 1"},
            }
        },
        object_info={
            "DynamicNode": {
                "input": {
                    "required": {
                        "expression": ["STRING", {}],
                        "values": ["STRING", {}],
                    }
                },
                "output": ["FLOAT"],
            }
        },
        filename="dynamic.json",
    )

    assert classes == ("DynamicNode",)


def test_validate_prompt_structure_finds_bad_output_slot_and_type() -> None:
    object_info = {
        "ImageSource": {
            "input": {"required": {}},
            "output": ["IMAGE"],
        },
        "ModelTarget": {
            "input": {"required": {"model": ["MODEL", {}]}},
            "output": [],
        },
    }

    with pytest.raises(ComfyE2EError, match="출력 슬롯 범위 오류"):
        _validate_prompt_structure(
            prompt={
                "1": {"class_type": "ImageSource", "inputs": {}},
                "2": {
                    "class_type": "ModelTarget",
                    "inputs": {"model": ["1", 1]},
                },
            },
            object_info=object_info,
            filename="bad-slot.json",
        )

    with pytest.raises(ComfyE2EError, match="연결 타입 불일치"):
        _validate_prompt_structure(
            prompt={
                "1": {"class_type": "ImageSource", "inputs": {}},
                "2": {
                    "class_type": "ModelTarget",
                    "inputs": {"model": ["1", 0]},
                },
            },
            object_info=object_info,
            filename="bad-type.json",
        )


def test_prepare_e2e_fixtures_creates_expected_images(tmp_path: Path) -> None:
    face_source = (
        tmp_path
        / "input"
        / "soya_char_ref"
        / "fallback"
        / "face.webp"
    )
    face_source.parent.mkdir(parents=True)
    Image.new("RGB", (96, 128), (12, 34, 56)).save(
        face_source,
        format="WEBP",
        lossless=True,
    )

    result = prepare_e2e_fixtures(tmp_path)
    assert set(result) == {
        "default",
        "face_tag",
        "training",
        "face",
        "edit_source",
        "edit_mask",
        "face_source",
    }
    for key, path in result.items():
        assert Path(path).is_file()
    assert Path(result["default"]).read_bytes()[:4] == b"RIFF"
    assert Path(result["face"]).name == "representation.png"
    assert Path(result["face_tag"]).name == "comfy-installer-e2e-face.png"
    assert Path(result["face_source"]) == face_source
    for key in ("face", "face_tag"):
        with Image.open(result[key]) as face_image:
            assert face_image.size == (96, 128)
            assert face_image.convert("RGB").getpixel((0, 0)) == (12, 34, 56)
    with Image.open(result["training"]) as training_image:
        assert training_image.size == (512, 512)


def test_protected_e2e_fixtures_restores_existing_and_removes_created(
    tmp_path: Path,
) -> None:
    comfy_root = tmp_path / "comfy"
    requirements_dir = tmp_path / "요구사항"
    default_image = comfy_root / "input" / "eri_default.webp"
    default_image.parent.mkdir(parents=True)
    original_default = b"original-user-image"
    default_image.write_bytes(original_default)
    face_source = (
        comfy_root
        / "input"
        / "soya_char_ref"
        / "fallback"
        / "face.png"
    )
    face_source.parent.mkdir(parents=True)
    Image.new("RGB", (64, 80), (90, 80, 70)).save(face_source)

    input_workspace = comfy_root / "input" / "comfy-installer-e2e"
    output_workspace = comfy_root / "output" / "comfy-installer-e2e"
    lora_workspace = (
        comfy_root
        / "models"
        / "loras"
        / "SOYA_CHAR_LORA"
        / "comfy-installer-e2e"
    )
    runtime_root = (
        comfy_root
        / "custom_nodes"
        / "comfyui-instant-lora_v_soya"
        / "runtime"
    )
    last_lora = runtime_root / "last_lora.json"
    existing_files = {
        input_workspace / "old.txt": b"original-input",
        output_workspace / "old.png": b"original-output",
        lora_workspace / "old.safetensors": b"original-lora",
        last_lora: b'{"path":"original"}',
    }
    for path, payload in existing_files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    tracked_roots = [
        runtime_root / "cache",
        runtime_root / "datasets",
        runtime_root / "artifacts",
    ]
    for root in tracked_roots:
        keep = root / "existing" / "keep.bin"
        keep.parent.mkdir(parents=True)
        keep.write_bytes(b"keep")

    with protected_e2e_fixtures(
        comfy_root=comfy_root,
        requirements_dir=requirements_dir,
    ) as fixtures:
        assert default_image.read_bytes() != original_default
        assert Path(fixtures["training"]).is_file()
        assert not (input_workspace / "old.txt").exists()
        assert not output_workspace.exists()
        assert not lora_workspace.exists()

        (input_workspace / "face" / "cache.pt").write_bytes(b"new-cache")
        (input_workspace / "training" / "sample.txt").write_text(
            "generated caption",
            encoding="utf-8",
        )
        output_workspace.mkdir(parents=True)
        (output_workspace / "new.png").write_bytes(b"new-output")
        lora_workspace.mkdir(parents=True)
        (lora_workspace / "new.safetensors").write_bytes(b"new-lora")
        last_lora.write_bytes(b'{"path":"generated"}')
        for root in tracked_roots:
            generated = root / "generated" / "artifact.bin"
            generated.parent.mkdir(parents=True)
            generated.write_bytes(b"generated")

    assert default_image.read_bytes() == original_default
    assert not Path(fixtures["training"]).exists()
    assert not Path(fixtures["face_tag"]).exists()
    for path, payload in existing_files.items():
        assert path.read_bytes() == payload
    assert not (input_workspace / "face" / "cache.pt").exists()
    assert not (input_workspace / "training" / "sample.txt").exists()
    assert not (output_workspace / "new.png").exists()
    assert not (lora_workspace / "new.safetensors").exists()
    for root in tracked_roots:
        assert (root / "existing" / "keep.bin").read_bytes() == b"keep"
        assert not (root / "generated").exists()

    backups = list(
        requirements_dir.glob(
            "comfy_e2e_fixture_before_*/input/eri_default.webp"
        )
    )
    assert len(backups) == 1
    assert backups[0].read_bytes() == original_default
    backup_root = backups[0].parents[1]
    assert (
        backup_root / "input" / "comfy-installer-e2e" / "old.txt"
    ).read_bytes() == b"original-input"
    assert face_source.is_file()


def test_promote_generated_fixture_preserves_face_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destinations = e2e_module._fixture_destinations(tmp_path)
    destinations["face"].parent.mkdir(parents=True)
    Image.new("RGB", (16, 16), (1, 2, 3)).save(destinations["face"])
    original_face = destinations["face"].read_bytes()
    Image.new("RGB", (16, 16), (4, 5, 6)).save(destinations["face_tag"])
    original_face_tag = destinations["face_tag"].read_bytes()

    payload_buffer = io.BytesIO()
    Image.new("RGB", (24, 24), (100, 110, 120)).save(
        payload_buffer,
        format="PNG",
    )
    response = SimpleNamespace(
        content=payload_buffer.getvalue(),
        raise_for_status=lambda: None,
    )
    monkeypatch.setattr(e2e_module.httpx, "get", lambda *args, **kwargs: response)

    promoted = promote_generated_fixture(
        base_url="http://127.0.0.1:8188",
        execution_result={
            "filename": "asset.json",
            "output_data": {
                "10": {
                    "images": [
                        {
                            "filename": "generated.png",
                            "subfolder": "",
                            "type": "output",
                        }
                    ]
                }
            },
        },
        comfy_root=tmp_path,
    )

    assert promoted == str(destinations["default"])
    assert destinations["face"].read_bytes() == original_face
    assert destinations["face_tag"].read_bytes() == original_face_tag
    for key in ("default", "training", "edit_source"):
        assert destinations[key].is_file()
        with Image.open(destinations[key]) as image:
            assert image.convert("RGB").getpixel((0, 0)) == (100, 110, 120)
