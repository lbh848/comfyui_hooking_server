import asyncio
import hashlib
import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modes import qwen_edit_mode as qwen_module
from modes.qwen_edit_mode import QwenEditMode
from queue_manager import (
    GPU_QUEUE_PRIORITY_TYPES,
    LLM_QUEUE_PRIORITY_TYPES,
    LLM_TYPES,
    QueueItem,
    QueueManager,
)


def _png_bytes(size=(320, 448), color=(50, 80, 120, 255)):
    output = io.BytesIO()
    Image.new("RGBA", size, color).save(output, format="PNG")
    return output.getvalue()


def _mask_bytes(size=(320, 448)):
    output = io.BytesIO()
    mask = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(mask)
    draw.ellipse((80, 100, 230, 330), fill=(255, 255, 255, 255))
    mask.save(output, format="PNG")
    return output.getvalue()


class FakeAssetMode:
    def __init__(self, source_path: Path):
        self.source_path = source_path

    def get_image_path(self, character, outfit, expression, filename):
        if (
            character == "alice"
            and outfit == "uniform"
            and expression == "smile"
            and filename == self.source_path.name
        ):
            return str(self.source_path)
        return None


def _configured_mode(tmp_path: Path):
    comfy_root = tmp_path / "ComfyUI"
    input_dir = comfy_root / "input"
    input_dir.mkdir(parents=True)
    checkpoint = (
        comfy_root
        / "models"
        / "checkpoints"
        / "v19"
        / "Qwen-Rapid-AIO-NSFW-v19.safetensors"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"test checkpoint placeholder")

    asset_dir = tmp_path / "asset" / "alice" / "uniform" / "smile"
    asset_dir.mkdir(parents=True)
    source_path = asset_dir / "source.webp"
    Image.new("RGB", (320, 448), (50, 80, 120)).save(
        source_path,
        format="WEBP",
        lossless=True,
    )
    source_prompt_path = asset_dir / "source_prompt.json"
    source_prompt_path.write_text(
        json.dumps(
            {
                "positive": "1girl, blue hair, school uniform",
                "negative": "low quality",
                "character": "alice",
                "outfit": "uniform",
                "expression": "smile",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    mode = QwenEditMode(asset_mode=FakeAssetMode(source_path))
    mode.get_config = lambda: {"comfy_input_dir": str(input_dir)}
    return mode, source_path, input_dir


def test_qwen_edit_stages_matching_source_and_mask_without_touching_asset(tmp_path):
    mode, source_path, input_dir = _configured_mode(tmp_path)
    original_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()

    staged = mode.stage_request(
        character="alice",
        outfit="uniform",
        expression="smile",
        filename="source.webp",
        mask_data=_mask_bytes(),
        edit_prompt="Replace the jacket with a black leather jacket.",
        edit_prompt_original="재킷을 검은 가죽 재킷으로 바꿔줘.",
        negative_prompt="blurry",
        seed=-1,
        steps=6,
        cfg=1,
        denoise=1,
        mask_grow=8,
        mask_blur=4,
    )

    assert not (input_dir / "qwen_edit").exists()
    pending = mode._pending_inputs[staged["job_id"]]
    with (
        Image.open(io.BytesIO(pending["source"])) as source,
        Image.open(io.BytesIO(pending["mask"])) as mask,
    ):
        assert source.size == mask.size
        assert source.size == (staged["width"], staged["height"])
        assert mask.convert("L").getbbox() is not None
    assert staged["image_path"] == "qwen_edit"
    assert staged["mask_path"] == "qwen_edit"
    assert staged["source_prompt"]["positive"].startswith("1girl")
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == original_hash
    assert list(source_path.parent.glob("*.webp")) == [source_path]


def test_qwen_edit_uses_uploaded_composite_source_without_overwriting_asset(tmp_path):
    mode, source_path, _input_dir = _configured_mode(tmp_path)
    original_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()

    staged = mode.stage_request(
        character="alice",
        outfit="uniform",
        expression="smile",
        filename="source.webp",
        mask_data=_mask_bytes(),
        source_data=_png_bytes(color=(210, 20, 40, 255)),
        edit_prompt="Integrate the composited item naturally.",
    )

    pending = mode._pending_inputs[staged["job_id"]]
    with Image.open(io.BytesIO(pending["source"])) as source:
        assert source.convert("RGB").getpixel((0, 0)) == (210, 20, 40)
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == original_hash


def test_qwen_edit_rejects_composite_source_with_wrong_dimensions(tmp_path):
    mode, _source_path, _input_dir = _configured_mode(tmp_path)

    with pytest.raises(ValueError, match="크기"):
        mode.stage_request(
            character="alice",
            outfit="uniform",
            expression="smile",
            filename="source.webp",
            mask_data=_mask_bytes(),
            source_data=_png_bytes(size=(64, 64)),
            edit_prompt="Integrate the composited item naturally.",
        )


def test_qwen_edit_rejects_empty_mask_with_diagnostic_input(tmp_path):
    mode, _source_path, _input_dir = _configured_mode(tmp_path)
    empty_mask = io.BytesIO()
    Image.new("RGBA", (320, 448), (0, 0, 0, 0)).save(
        empty_mask,
        format="PNG",
    )

    with pytest.raises(ValueError, match="마스크"):
        mode.stage_request(
            character="alice",
            outfit="uniform",
            expression="smile",
            filename="source.webp",
            mask_data=empty_mask.getvalue(),
            edit_prompt="Change the masked area.",
        )


def test_qwen_edit_rejects_blank_comfy_input_setting(tmp_path):
    mode, source_path, _input_dir = _configured_mode(tmp_path)
    mode.asset_mode = FakeAssetMode(source_path)
    mode.get_config = lambda: {"comfy_input_dir": "   "}

    with pytest.raises(ValueError, match="Comfy input"):
        mode.stage_request(
            character="alice",
            outfit="uniform",
            expression="smile",
            filename="source.webp",
            mask_data=_mask_bytes(),
            edit_prompt="Change the masked area.",
        )


@pytest.mark.asyncio
async def test_qwen_edit_loads_selected_ui_workflow_through_converter(tmp_path):
    mode, _source_path, _input_dir = _configured_mode(tmp_path)
    selected_workflow = tmp_path / "selected_qwen_edit_ui.json"
    selected_workflow.write_text(
        json.dumps(
            {
                "last_node_id": 1,
                "last_link_id": 0,
                "nodes": [{"id": 1, "type": "TestNode"}],
                "links": [],
                "version": 0.4,
            }
        ),
        encoding="utf-8",
    )
    expected_api = json.loads(
        (ROOT / "mode_workflow" / "배포_qwen_edit_v1.json").read_text(
            encoding="utf-8"
        )
    )
    converted = []

    async def convert(workflow):
        converted.append(workflow)
        return expected_api, None

    mode.convert_workflow_func = convert
    workflow, workflow_path = await mode._load_workflow(
        {"qwen_edit_workflow_source_path": str(selected_workflow)}
    )

    assert converted[0]["nodes"][0]["type"] == "TestNode"
    assert workflow == expected_api
    assert workflow_path == str(selected_workflow.resolve())


@pytest.mark.asyncio
async def test_qwen_edit_mocked_comfy_e2e_appends_result_and_metadata(tmp_path):
    mode, source_path, input_dir = _configured_mode(tmp_path)
    original_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    qwen_input = input_dir / "qwen_edit"
    qwen_input.mkdir()
    (qwen_input / "stale.png").write_bytes(_png_bytes())
    stale_dir = qwen_input / "old_job"
    stale_dir.mkdir()
    (stale_dir / "old.png").write_bytes(_png_bytes())
    staged = mode.stage_request(
        character="alice",
        outfit="uniform",
        expression="smile",
        filename="source.webp",
        mask_data=_mask_bytes(),
        edit_prompt="Replace the jacket with a black leather jacket.",
        edit_prompt_original="재킷을 검은 가죽 재킷으로 바꿔줘.",
        negative_prompt="blurry",
        seed=123,
        steps=6,
        cfg=1,
        denoise=1,
        mask_grow=8,
        mask_blur=4,
    )
    assert (qwen_input / "stale.png").is_file()
    assert stale_dir.is_dir()
    submitted = {}
    notifications = []

    async def submit(workflow, progress_callback=None):
        submitted.update(workflow)
        prompt_node = next(
            node
            for node in workflow.values()
            if node.get("class_type") == "PrimitiveStringMultiline"
            and node.get("_meta", {}).get("title") == "긍정프롬프트"
        )
        payload = prompt_node["inputs"]["value"]
        assert "[EDIT_PROMPT]\nReplace the jacket" in payload
        assert f"[IMAGE_PATH]\n{staged['image_path']}" in payload
        assert f"[MASK_PATH]\n{staged['mask_path']}" in payload
        assert f"[WIDTH]\n{staged['width']}" in payload
        assert f"[HEIGHT]\n{staged['height']}" in payload
        assert workflow["2"]["inputs"]["text"] == ["1", 0]
        assert workflow["4"]["inputs"]["path"] == ["2", 2]
        assert sorted(path.name for path in qwen_input.iterdir()) == [
            "mask.png",
            "source.png",
        ]
        with (
            Image.open(qwen_input / "source.png") as source,
            Image.open(qwen_input / "mask.png") as mask,
        ):
            assert source.size == mask.size
            assert mask.convert("L").getbbox() is not None
        if progress_callback:
            await progress_callback(3, 6)
            await progress_callback(6, 6)
        return _png_bytes(), None

    async def notify(event_type, data):
        notifications.append((event_type, data))

    mode.submit_workflow_func = submit
    mode.notify_frontend_func = notify
    result = await mode.execute(staged)

    assert result["success"] is True
    assert result["filename"] != source_path.name
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == original_hash
    result_path = source_path.parent / result["filename"]
    assert result_path.is_file()
    assert len(list(source_path.parent.glob("*.webp"))) == 2
    record = json.loads(
        result_path.with_name(f"{result_path.stem}_prompt.json").read_text(
            encoding="utf-8"
        )
    )
    assert record["positive"] == "1girl, blue hair, school uniform"
    assert record["negative"] == "low quality"
    assert record["is_edited"] is True
    assert record["edit_prompt"].startswith("Replace the jacket")
    assert record["edit_prompt_original"].startswith("재킷")
    assert record["edit_source_filename"] == "source.webp"
    assert record["edit_seed"] == 123
    assert not any(
        node["class_type"] in {"ImageCompositeMasked", "EmptyLatentImage"}
        for node in submitted.values()
    )
    assert submitted["4"]["class_type"] == "LoadImagesFromPath_mdsoya"
    assert submitted["5"]["class_type"] == "ImageFromBatch"
    assert submitted["5"]["inputs"]["batch_index"] == 1
    assert submitted["6"]["inputs"]["batch_index"] == 0
    assert submitted["9"]["class_type"] == "VAEEncode"
    assert submitted["10"]["class_type"] == "SetLatentNoiseMask"
    assert submitted["10"]["inputs"]["samples"] == ["9", 0]
    assert submitted["10"]["inputs"]["mask"] == ["8", 0]
    assert submitted["11"]["class_type"] == "SoyaQwenEditConditioning_mdsoya"
    assert submitted["11"]["inputs"]["target_latent"] == ["10", 0]
    assert submitted["13"]["inputs"]["latent_image"] == ["10", 0]
    assert submitted["13"]["inputs"]["sampler_name"] == "er_sde"
    assert submitted["13"]["inputs"]["scheduler"] == "beta"
    assert submitted["15"]["inputs"]["images"] == ["14", 0]
    assert any(event == "qwen_edit_completed" for event, _data in notifications)
    mode.cleanup_staged_request(staged)
    assert staged["job_id"] not in mode._pending_inputs


@pytest.mark.asyncio
async def test_qwen_edit_uses_gpu_queue_and_translation_uses_llm_queue():
    assert "qwen_edit" in GPU_QUEUE_PRIORITY_TYPES
    assert "qwen_edit_translate" in LLM_QUEUE_PRIORITY_TYPES
    assert "qwen_edit" not in LLM_TYPES
    assert "qwen_edit_translate" in LLM_TYPES

    manager = QueueManager()
    calls = []

    class FakeQwenMode:
        async def execute(self, params, progress_callback=None):
            calls.append(("edit", params))
            if progress_callback:
                await progress_callback(6, 6)
            return {"success": True, "filename": "edited.webp"}

        async def translate_prompt(
            self,
            text,
            queue_item_id="",
            *,
            edit_tool="qwen",
            source_prompt="",
        ):
            calls.append(
                ("translate", text, queue_item_id, edit_tool, source_prompt)
            )
            return {"success": True, "translated_prompt": "Change the jacket."}

        def cleanup_staged_request(self, params):
            calls.append(("cleanup", params.get("job_id")))

    manager.qwen_edit_mode = FakeQwenMode()
    manager.notify_frontend = lambda *_args, **_kwargs: asyncio.sleep(0)
    edit_item = QueueItem(
        id="qwen-edit-1",
        type="qwen_edit",
        label="edit",
        params={"job_id": "job", "source_filename": "source.webp"},
    )
    translate_item = QueueItem(
        id="qwen-translate-1",
        type="qwen_edit_translate",
        label="translate",
        params={"text": "재킷을 바꿔줘"},
    )

    assert (await manager._handle_qwen_edit(edit_item))["success"] is True
    translated = await manager._handle_qwen_edit_translate(translate_item)
    assert translated["translated_prompt"] == "Change the jacket."
    assert calls[0][0] == "edit"
    assert calls[1] == ("cleanup", "job")
    assert calls[2] == (
        "translate",
        "재킷을 바꿔줘",
        "qwen-translate-1",
        "qwen",
        "",
    )


def test_qwen_parser_node_contract_and_errors():
    parser_path = (
        Path(r"E:\wsl2\matrix\Packages\ComfyUI\custom_nodes")
        / "comfyui-soya-custom-nodes"
        / "soya_qwen_edit_prompt_parser.py"
    )
    spec = importlib.util.spec_from_file_location("soya_qwen_parser_test", parser_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    parser = module.SoyaQwenEditPromptParser_mdsoya()

    values = parser.parse(
        "[EDIT_PROMPT]\nChange the jacket.\n"
        "[NEGATIVE_PROMPT]\nblurry\n"
        "[IMAGE_PATH]\nqwen_edit/job/source.png\n"
        "[MASK_PATH]\nqwen_edit/job/mask.png\n"
        "[SEED]\n123\n"
        "[STEPS]\n6\n"
        "[CFG]\n1\n"
        "[DENOISE]\n1\n"
        "[MASK_GROW]\n8\n"
        "[MASK_BLUR]\n4\n"
        "[FILENAME_PREFIX]\nqwen_edit/job/output\n"
        "[WIDTH]\n1024\n"
        "[HEIGHT]\n768"
    )
    assert values[0] == "Change the jacket."
    assert values[2].endswith("source.png")
    assert values[4:10] == (123, 6, 1.0, 1.0, 8, 4.0)
    assert values[11:13] == (1024, 768)
    with pytest.raises(ValueError, match="비어"):
        parser.parse("")


def test_qwen_workflow_and_frontend_contracts():
    workflow = json.loads(
        (ROOT / "mode_workflow" / "배포_qwen_edit_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert workflow["3"]["inputs"]["ckpt_name"].endswith(
        "Qwen-Rapid-AIO-NSFW-v19.safetensors"
    )
    assert workflow["1"]["class_type"] == "PrimitiveStringMultiline"
    assert workflow["1"]["_meta"]["title"] == "긍정프롬프트"
    assert workflow["2"]["inputs"]["text"] == ["1", 0]
    assert workflow["4"]["class_type"] == "LoadImagesFromPath_mdsoya"
    assert workflow["4"]["inputs"]["path"] == ["2", 2]
    assert workflow["5"]["inputs"]["batch_index"] == 1
    assert workflow["6"]["inputs"]["batch_index"] == 0
    assert workflow["9"]["class_type"] == "VAEEncode"
    assert workflow["10"]["class_type"] == "SetLatentNoiseMask"
    assert workflow["11"]["class_type"] == "SoyaQwenEditConditioning_mdsoya"
    assert workflow["11"]["inputs"]["target_latent"] == ["10", 0]
    assert workflow["13"]["inputs"]["latent_image"] == ["10", 0]
    assert workflow["15"]["inputs"]["images"] == ["14", 0]
    assert workflow["13"]["inputs"]["steps"] == ["2", 5]
    assert workflow["13"]["inputs"]["cfg"] == ["2", 6]
    assert not any(
        node["class_type"] in {"ImageCompositeMasked", "EmptyLatentImage"}
        for node in workflow.values()
    )
    assert "[WIDTH]\n1024" in workflow["1"]["inputs"]["value"]
    assert "[HEIGHT]\n1024" in workflow["1"]["inputs"]["value"]

    ui_workflow = json.loads(
        (ROOT / "mode_workflow" / "배포_qwen_edit_v1_변환전.json").read_text(
            encoding="utf-8"
        )
    )
    assert isinstance(ui_workflow["nodes"], list)
    assert isinstance(ui_workflow["links"], list)
    assert len(ui_workflow["nodes"]) == 15
    ui_by_id = {str(node["id"]): node for node in ui_workflow["nodes"]}
    assert ui_by_id["1"]["title"] == "긍정프롬프트"
    assert ui_by_id["4"]["type"] == "LoadImagesFromPath_mdsoya"
    assert ui_by_id["10"]["type"] == "SetLatentNoiseMask"

    frontend = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    assert "QWEN EDIT" in frontend
    assert "EDIT됨" in frontend
    assert "function auOpenQwenEdit(data)" in frontend
    assert "function auQwenTranslatePrompt()" in frontend
    assert "function auQwenEnqueueEdit()" in frontend
    assert "function auQwenBuildCompositePayload(state)" in frontend
    assert 'id="au-qwen-composite-add-btn"' in frontend
    assert 'id="au-qwen-composite-tool-btn"' in frontend
    assert 'id="au-qwen-item-crop-btn"' in frontend
    assert 'id="au-qwen-item-crop-apply-btn"' in frontend
    assert frontend.count("event.button !== 1") >= 2
    assert "middlePanning" in frontend
    assert "마우스 휠 버튼 드래그로 화면을 이동" in frontend
    assert "asset_data/qwen_composite_items" in frontend
    assert "/api/asset_mode/qwen_edit/translate" in frontend
    assert "/api/asset_mode/qwen_edit/enqueue" in frontend
    assert "/api/asset_mode/qwen_edit/composite_items" in frontend
    assert "qwen_edit_translate: 'EDIT 번역'" in frontend
    assert "{types: ['qwen_edit'], label: 'EDIT 툴 마스크 편집'}" in frontend
    assert "선택된 EDIT 툴의 latent 샘플링 마스크" in frontend
    assert "반투명 파란색으로 칠한 영역" in frontend
    assert "function auQwenFitMaskCanvas()" in frontend
    assert "ctx.strokeStyle = '#168cff'" in frontend
    assert 'id="au-qwen-mask-opacity"' in frontend
    assert "const updateMaskDisplayOpacity = () =>" in frontend
    assert "overflow: hidden;" in frontend[
        frontend.index(".qwen-mask-stage {"):
        frontend.index(".qwen-mask-canvas-wrap {")
    ]
    assert 'id="setting-qwen-edit-workflow-filename"' in frontend
    assert 'id="setting-qwen-edit-workflow-source-path"' in frontend
    assert 'id="setting-asset-edit-tool"' in frontend
    assert 'value="anima_inpainting"' in frontend
    assert 'id="setting-anima-inpainting-workflow-filename"' in frontend
    assert 'id="setting-anima-inpainting-workflow-source-path"' in frontend
    assert "배포_ANIMA_inpainting_v1.json" in frontend
    assert "qwen_edit_workflow_source_path:" in frontend
    assert "anima_inpainting_workflow_source_path:" in frontend
    assert "asset_edit_tool:" in frontend
    upload_lv2 = frontend[
        frontend.index("async function auRenderImages("):
        frontend.index("let auQwenEditState")
    ]
    asset_generation_lv2 = frontend[
        frontend.index("async function loadAssetImages()"):
        frontend.index("async function setAssetRepresentative(")
    ]
    assert "EDIT 툴 호출" in upload_lv2
    assert "EDIT 툴 호출" in asset_generation_lv2
    assert "auOpenQwenEdit({" in asset_generation_lv2
    assert "EDIT됨" in asset_generation_lv2
    assert "<b>Edit:</b>" in asset_generation_lv2

    server_source = (ROOT / "server.py").read_text(encoding="utf-8")
    assert "handle_api_qwen_edit_translate" in server_source
    assert "handle_api_qwen_edit_enqueue" in server_source
    assert "handle_api_qwen_composite_item_upload" in server_source
    assert "handle_api_qwen_composite_background_remove" in server_source
    assert "handle_api_qwen_composite_item_delete" in server_source
    assert 'queue_manager.add_item(\n            "qwen_edit"' in server_source
    assert '"qwen_edit_workflow_source_path": os.path.join(' in server_source
    assert "qwen_edit_mode.convert_workflow_func = lambda workflow:" in server_source
    assert 'task_key="qwen_edit"' in server_source
