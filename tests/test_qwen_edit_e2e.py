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

    job_dir = input_dir / "qwen_edit" / staged["job_id"]
    source_stage = job_dir / "source.png"
    mask_stage = job_dir / "mask.png"
    assert source_stage.is_file()
    assert mask_stage.is_file()
    with Image.open(source_stage) as source, Image.open(mask_stage) as mask:
        assert source.size == mask.size
        assert source.size == (staged["width"], staged["height"])
        assert mask.convert("L").getbbox() is not None
    assert staged["image_path"].endswith("/source.png")
    assert staged["mask_path"].endswith("/mask.png")
    assert staged["source_prompt"]["positive"].startswith("1girl")
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == original_hash
    assert list(source_path.parent.glob("*.webp")) == [source_path]


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
async def test_qwen_edit_mocked_comfy_e2e_appends_result_and_metadata(tmp_path):
    mode, source_path, _input_dir = _configured_mode(tmp_path)
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
        seed=123,
        steps=6,
        cfg=1,
        denoise=1,
        mask_grow=8,
        mask_blur=4,
    )
    submitted = {}
    notifications = []

    async def submit(workflow, progress_callback=None):
        submitted.update(workflow)
        parser = next(
            node
            for node in workflow.values()
            if node.get("class_type") == "SoyaQwenEditPromptParser_mdsoya"
        )
        assert "[EDIT_PROMPT]\nReplace the jacket" in parser["inputs"]["text"]
        assert f"[IMAGE_PATH]\n{staged['image_path']}" in parser["inputs"]["text"]
        assert f"[MASK_PATH]\n{staged['mask_path']}" in parser["inputs"]["text"]
        assert f"[WIDTH]\n{staged['width']}" in parser["inputs"]["text"]
        assert f"[HEIGHT]\n{staged['height']}" in parser["inputs"]["text"]
        assert workflow["3"]["inputs"]["image"] == staged["image_path"]
        assert workflow["4"]["inputs"]["image"] == staged["mask_path"]
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
    assert submitted["8"]["class_type"] == "ImageCompositeMasked"
    assert submitted["8"]["inputs"]["destination"] == ["3", 0]
    assert submitted["8"]["inputs"]["source"] == ["12", 0]
    assert submitted["8"]["inputs"]["mask"] == ["6", 0]
    assert submitted["9"]["class_type"] == "SoyaQwenEditConditioning_mdsoya"
    assert submitted["7"]["class_type"] == "EmptyLatentImage"
    assert submitted["7"]["inputs"]["width"] == ["1", 11]
    assert submitted["7"]["inputs"]["height"] == ["1", 12]
    assert submitted["9"]["inputs"]["target_latent"] == ["7", 0]
    assert submitted["11"]["inputs"]["latent_image"] == ["7", 0]
    assert submitted["11"]["inputs"]["sampler_name"] == "er_sde"
    assert submitted["11"]["inputs"]["scheduler"] == "beta"
    assert any(event == "qwen_edit_completed" for event, _data in notifications)


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

        async def translate_prompt(self, text, queue_item_id=""):
            calls.append(("translate", text, queue_item_id))
            return {"success": True, "translated_prompt": "Change the jacket."}

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
    assert calls[1] == ("translate", "재킷을 바꿔줘", "qwen-translate-1")


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
    assert workflow["2"]["inputs"]["ckpt_name"].endswith(
        "Qwen-Rapid-AIO-NSFW-v19.safetensors"
    )
    assert workflow["8"]["class_type"] == "ImageCompositeMasked"
    assert workflow["7"]["class_type"] == "EmptyLatentImage"
    assert workflow["9"]["class_type"] == "SoyaQwenEditConditioning_mdsoya"
    assert workflow["9"]["inputs"]["target_latent"] == ["7", 0]
    assert workflow["13"]["inputs"]["images"] == ["8", 0]
    assert workflow["11"]["inputs"]["steps"] == ["1", 5]
    assert workflow["11"]["inputs"]["cfg"] == ["1", 6]
    assert "[WIDTH]\n1024" in workflow["1"]["inputs"]["text"]
    assert "[HEIGHT]\n1024" in workflow["1"]["inputs"]["text"]

    frontend = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    assert "QWEN EDIT" in frontend
    assert "EDIT됨" in frontend
    assert "function auOpenQwenEdit(data)" in frontend
    assert "function auQwenTranslatePrompt()" in frontend
    assert "function auQwenEnqueueEdit()" in frontend
    assert "/api/asset_mode/qwen_edit/translate" in frontend
    assert "/api/asset_mode/qwen_edit/enqueue" in frontend
    assert "qwen_edit_translate: 'Qwen 번역'" in frontend
    assert "{types: ['qwen_edit'], label: 'Qwen 마스크 편집'}" in frontend

    server_source = (ROOT / "server.py").read_text(encoding="utf-8")
    assert "handle_api_qwen_edit_translate" in server_source
    assert "handle_api_qwen_edit_enqueue" in server_source
    assert 'queue_manager.add_item(\n            "qwen_edit"' in server_source
