from __future__ import annotations

from pathlib import Path
from threading import Event

import pytest

from comfy_installer.e2e import (
    ComfyE2EError,
    ComfyProcess,
    WorkflowValidation,
    _build_prompt_request,
    _validate_prompt_structure,
    bypass_sageattention_nodes,
    make_e2e_prompt,
    prepare_e2e_fixtures,
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


def test_prepare_e2e_fixtures_creates_expected_images(tmp_path: Path) -> None:
    result = prepare_e2e_fixtures(tmp_path)
    assert set(result) == {
        "default",
        "training",
        "face",
        "edit_source",
        "edit_mask",
    }
    for path in result.values():
        assert Path(path).is_file()
    assert Path(result["default"]).read_bytes()[:4] == b"RIFF"
    assert Path(result["face"]).name == "representation.png"


def test_protected_e2e_fixtures_restores_existing_and_removes_created(
    tmp_path: Path,
) -> None:
    comfy_root = tmp_path / "comfy"
    requirements_dir = tmp_path / "요구사항"
    existing = comfy_root / "input" / "eri_default.webp"
    existing.parent.mkdir(parents=True)
    original = b"original-user-image"
    existing.write_bytes(original)

    with protected_e2e_fixtures(
        comfy_root=comfy_root,
        requirements_dir=requirements_dir,
    ) as fixtures:
        assert existing.read_bytes() != original
        assert Path(fixtures["training"]).is_file()

    assert existing.read_bytes() == original
    assert not Path(fixtures["training"]).exists()
    backups = list(
        requirements_dir.glob(
            "comfy_e2e_fixture_before_*/input/eri_default.webp"
        )
    )
    assert len(backups) == 1
    assert backups[0].read_bytes() == original
