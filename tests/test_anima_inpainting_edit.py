import io
import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from modes import qwen_edit_mode as qwen_module
from modes.qwen_edit_mode import (
    ANIMA_INPAINTING_LLLITE_FILENAME,
    QwenEditMode,
)


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = ROOT / "mode_workflow"
SOYA_DIR = Path(
    r"E:\wsl2\matrix\Packages\ComfyUI\custom_nodes"
) / "comfyui-soya-custom-nodes"


def test_anima_workflow_uses_exact_deployment_name_and_has_api_companion():
    selected_path = WORKFLOW_DIR / "배포_ANIMA_inpainting_v1.json"
    api_path = WORKFLOW_DIR / "배포_ANIMA_inpainting_v1_API.json"

    assert selected_path.is_file()
    assert api_path.is_file()
    assert not (WORKFLOW_DIR / "배포_ANIMA_inpainting_v1_변환전.json").exists()

    workflow = json.loads(selected_path.read_text(encoding="utf-8"))
    assert isinstance(workflow["nodes"], list)
    assert isinstance(workflow["links"], list)
    assert len(workflow["nodes"]) == 18

    nodes = {str(node["id"]): node for node in workflow["nodes"]}
    assert nodes["3"]["type"] == "UNETLoader"
    assert nodes["3"]["widgets_values"][0] == "anima_baseV10.safetensors"
    assert nodes["6"]["title"] == "ANIMA Inpainting 입력 폴더"
    assert nodes["15"]["type"] == "SoyaAnimaLLLiteApply_mdsoya"
    assert nodes["15"]["widgets_values"] == [
        ANIMA_INPAINTING_LLLITE_FILENAME,
        1,
        0,
        1,
        True,
    ]
    assert nodes["16"]["widgets_values"][2:] == [
        30,
        4,
        "er_sde",
        "simple",
        0.8,
    ]


def test_anima_api_workflow_preserves_source_and_mask_inpainting_contract():
    workflow = json.loads(
        (
            WORKFLOW_DIR / "배포_ANIMA_inpainting_v1_API.json"
        ).read_text(encoding="utf-8")
    )

    assert workflow["7"]["inputs"]["batch_index"] == 1
    assert workflow["8"]["inputs"]["batch_index"] == 0
    assert workflow["11"]["inputs"]["pixels"] == ["7", 0]
    assert workflow["12"]["inputs"]["samples"] == ["11", 0]
    assert workflow["12"]["inputs"]["mask"] == ["10", 0]

    lllite = workflow["15"]
    assert lllite["class_type"] == "SoyaAnimaLLLiteApply_mdsoya"
    assert lllite["inputs"]["model"] == ["3", 0]
    assert lllite["inputs"]["image"] == ["7", 0]
    assert lllite["inputs"]["mask"] == ["10", 0]
    assert lllite["inputs"]["lllite_name"] == ANIMA_INPAINTING_LLLITE_FILENAME
    assert lllite["inputs"]["preserve_wrapper"] is True

    sampler = workflow["16"]["inputs"]
    assert sampler["model"] == ["15", 0]
    assert sampler["latent_image"] == ["12", 0]
    assert sampler["sampler_name"] == "er_sde"
    assert sampler["scheduler"] == "simple"


def test_anima_edit_keeps_supported_high_resolution_while_qwen_still_caps_it():
    qwen_limits = QwenEditMode._size_limits("qwen")
    anima_limits = QwenEditMode._size_limits("anima_inpainting")

    assert QwenEditMode._target_size(
        1536,
        1024,
        max_pixels=qwen_limits[0],
        max_edge=qwen_limits[1],
    ) == (1248, 832)
    assert QwenEditMode._target_size(
        1536,
        1024,
        max_pixels=anima_limits[0],
        max_edge=anima_limits[1],
    ) == (1536, 1024)


def test_anima_model_path_is_controlnet_and_payload_has_separate_output_root(
    tmp_path,
):
    input_dir = tmp_path / "ComfyUI" / "input"
    input_dir.mkdir(parents=True)

    model_path = QwenEditMode._required_model_path(
        {"comfy_input_dir": str(input_dir)},
        "anima_inpainting",
    )
    assert Path(model_path) == (
        tmp_path
        / "ComfyUI"
        / "models"
        / "controlnet"
        / ANIMA_INPAINTING_LLLITE_FILENAME
    )

    payload = QwenEditMode._build_parser_payload(
        {
            "edit_tool": "anima_inpainting",
            "edit_prompt": "black leather jacket",
            "negative_prompt": "blurry",
            "image_path": "qwen_edit/job",
            "mask_path": "qwen_edit/job",
            "seed": 1,
            "steps": 30,
            "cfg": 4,
            "denoise": 0.8,
            "mask_grow": 8,
            "mask_blur": 4,
            "job_id": "job",
            "width": 1536,
            "height": 1024,
        }
    )
    assert "[FILENAME_PREFIX]\nanima_inpainting/job/output" in payload


def test_soya_contains_local_anima_lllite_implementation_and_registration():
    node_source = (SOYA_DIR / "soya_anima_lllite.py").read_text(
        encoding="utf-8"
    )
    core_source = (SOYA_DIR / "soya_anima_lllite_core.py").read_text(
        encoding="utf-8"
    )
    init_source = (SOYA_DIR / "__init__.py").read_text(encoding="utf-8")
    notice = (
        SOYA_DIR / "third_party" / "ComfyUI-Anima-LLLite-NOTICE.md"
    ).read_text(encoding="utf-8")

    assert "from .soya_anima_lllite_core import" in node_source
    assert "traceback.print_exc()" in node_source
    assert "class SoyaAnimaLLLiteApply_mdsoya" in node_source
    assert "class ControlNetLLLiteDiT" in core_source
    assert "ComfyUI-Anima-LLLite" not in node_source
    assert "SoyaAnimaLLLiteApply_mdsoya" in init_source
    assert "6701c8d6fc3bf1b2ed966b87c95ce609c52cebea" in notice


def test_global_asset_setting_waits_for_installed_library_workflow_binding():
    server_source = (ROOT / "server.py").read_text(encoding="utf-8")
    frontend = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")

    assert '"asset_edit_tool": "qwen"' in server_source
    assert '"anima_inpainting_workflow_source_path": ""' in server_source
    assert 'id="setting-asset-edit-tool"' in frontend
    assert 'value="anima_inpainting"' in frontend
    assert 'placeholder="라이브러리 배포 워크플로우"' in frontend


@pytest.mark.asyncio
async def test_anima_edit_mocked_comfy_e2e_uses_selected_ui_workflow(tmp_path):
    comfy_root = tmp_path / "ComfyUI"
    input_dir = comfy_root / "input"
    input_dir.mkdir(parents=True)
    weights = (
        comfy_root
        / "models"
        / "controlnet"
        / ANIMA_INPAINTING_LLLITE_FILENAME
    )
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"test weights placeholder")

    asset_dir = tmp_path / "asset" / "alice" / "uniform" / "smile"
    asset_dir.mkdir(parents=True)
    source_path = asset_dir / "source.webp"
    Image.new("RGB", (1536, 1024), (60, 90, 120)).save(
        source_path,
        format="WEBP",
        lossless=True,
    )
    source_path.with_name("source_prompt.json").write_text(
        json.dumps(
            {
                "positive": "1girl, school uniform",
                "negative": "low quality",
            }
        ),
        encoding="utf-8",
    )

    class AssetMode:
        @staticmethod
        def get_image_path(character, outfit, expression, filename):
            if (character, outfit, expression, filename) == (
                "alice",
                "uniform",
                "smile",
                "source.webp",
            ):
                return str(source_path)
            return None

    selected_ui = WORKFLOW_DIR / "배포_ANIMA_inpainting_v1.json"
    expected_api = json.loads(
        (
            WORKFLOW_DIR / "배포_ANIMA_inpainting_v1_API.json"
        ).read_text(encoding="utf-8")
    )
    mode = QwenEditMode(asset_mode=AssetMode())
    mode.get_config = lambda: {
        "asset_edit_tool": "anima_inpainting",
        "comfy_input_dir": str(input_dir),
        "anima_inpainting_workflow_source_path": str(selected_ui),
    }

    mask_buffer = io.BytesIO()
    mask = Image.new("RGBA", (1536, 1024), (0, 0, 0, 0))
    ImageDraw.Draw(mask).rectangle(
        (450, 250, 1050, 800),
        fill=(255, 255, 255, 255),
    )
    mask.save(mask_buffer, format="PNG")

    staged = mode.stage_request(
        character="alice",
        outfit="uniform",
        expression="smile",
        filename="source.webp",
        mask_data=mask_buffer.getvalue(),
        edit_prompt="black leather jacket, natural folds",
        edit_prompt_original="검은 가죽 재킷",
        negative_prompt="blurry",
        seed=123,
        steps=30,
        cfg=4,
        denoise=0.8,
        mask_grow=8,
        mask_blur=4,
    )
    assert staged["edit_tool"] == "anima_inpainting"
    assert (staged["width"], staged["height"]) == (1536, 1024)

    converted = []
    submitted = []

    async def convert(workflow):
        converted.append(workflow)
        return expected_api, None

    async def submit(workflow, progress_callback=None):
        submitted.append(workflow)
        if progress_callback:
            await progress_callback(30, 30)
        output = io.BytesIO()
        Image.new("RGB", (1536, 1024), (80, 110, 140)).save(
            output,
            format="PNG",
        )
        return output.getvalue(), None

    mode.convert_workflow_func = convert
    mode.submit_workflow_func = submit
    result = await mode.execute(staged)

    assert converted[0]["nodes"][14]["type"] == "SoyaAnimaLLLiteApply_mdsoya"
    assert submitted[0]["15"]["inputs"]["mask"] == ["10", 0]
    assert submitted[0]["16"]["inputs"]["scheduler"] == "simple"
    assert result["success"] is True
    assert result["edit_tool"] == "anima_inpainting"
    assert "_anima_inpaint_" in result["filename"]
    record = json.loads(
        Path(result["local_path"]).with_name(
            f"{Path(result['local_path']).stem}_prompt.json"
        ).read_text(encoding="utf-8")
    )
    assert record["edit_tool"] == "anima_inpainting"
    assert record["edit_model"] == "anima-lllite-inpainting-v2"
    mode.cleanup_staged_request(staged)


@pytest.mark.asyncio
async def test_anima_translation_requests_complete_final_image_prompt(monkeypatch):
    captured = {}

    async def fake_call(
        task_key,
        messages,
        *,
        metadata_sink=None,
        result_validator=None,
    ):
        captured["task_key"] = task_key
        captured["messages"] = messages
        if metadata_sink is not None:
            metadata_sink.update({"prompt_tokens": 20, "completion_tokens": 10})
        return (
            "masterpiece, highres, black-haired boy, "
            "bright red velvet jacket, white background"
        )

    monkeypatch.setattr(qwen_module.llm_service, "callLLMTask", fake_call)
    monkeypatch.setattr(
        qwen_module,
        "_log_lighbd_history",
        lambda _record: None,
    )
    mode = QwenEditMode()
    result = await mode.translate_prompt(
        "검은 재킷을 붉은 벨벳 재킷으로 바꿔줘",
        "anima-translation-test",
        edit_tool="anima_inpainting",
        source_prompt=(
            "black-haired boy, black jacket, white background, "
            "highres\n[PIPELINE_SETTING]\nignored"
        ),
    )

    system_prompt = captured["messages"][0]["content"]
    assert "complete English positive prompt" in system_prompt
    assert "entire intended image" in system_prompt
    assert "not only the masked region" in system_prompt
    assert "replace conflicting source attributes" in system_prompt
    assert "non-visual metadata" in system_prompt
    assert captured["task_key"] == "qwen_edit_translate"
    assert result["edit_tool"] == "anima_inpainting"
    assert "bright red velvet jacket" in result["translated_prompt"]
