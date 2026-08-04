import io
import json
from pathlib import Path
import sys

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modes import postprocess
import importlib
from queue_manager import QueueItem, QueueManager


bot_mode_module = importlib.import_module("modes.bot_mode")


def _png_bytes(color=(20, 40, 60)):
    output = io.BytesIO()
    Image.new("RGB", (48, 64), color).save(output, format="PNG")
    return output.getvalue()


def test_dialogue_face_crop_path_uses_face_crop_folder_and_suffix(tmp_path, monkeypatch):
    monkeypatch.setattr(bot_mode_module, "BOT_DIR", str(tmp_path / "bot"))

    path = Path(bot_mode_module.dialogue_face_crop_path(
        "sample-bot", "alice", "alice_happy.webp"
    ))

    assert path == tmp_path / "bot" / "sample-bot" / "alice" / "FACE CROP" / "alice_happy_face.png"


@pytest.mark.asyncio
async def test_dialogue_face_crop_overwrites_existing_and_continues_after_one_failure(
    tmp_path, monkeypatch
):
    bot_root = tmp_path / "bot"
    char_dir = bot_root / "sample-bot" / "alice"
    crop_dir = char_dir / "FACE CROP"
    comfy_input = tmp_path / "comfy-input"
    char_dir.mkdir(parents=True)
    crop_dir.mkdir()
    comfy_input.mkdir()
    (char_dir / "a.png").write_bytes(_png_bytes((255, 0, 0)))
    (char_dir / "b.png").write_bytes(_png_bytes((0, 255, 0)))
    (char_dir / "c.png").write_bytes(_png_bytes((0, 0, 255)))
    existing_bytes = _png_bytes((100, 100, 100))
    failed_existing_bytes = _png_bytes((120, 110, 100))
    (crop_dir / "a_face.png").write_bytes(existing_bytes)
    (crop_dir / "b_face.png").write_bytes(failed_existing_bytes)

    workflow_path = tmp_path / "face-workflow.json"
    workflow_path.write_text(json.dumps({"nodes": []}), encoding="utf-8")
    monkeypatch.setattr(bot_mode_module, "BOT_DIR", str(bot_root))

    manager = QueueManager()
    manager.get_config = lambda: {
        "comfy_input_dir": str(comfy_input),
        "face_extract_workflow_source_path": str(workflow_path),
    }

    async def convert_workflow(_raw, task_key):
        assert task_key == "face_extract"
        return {
            "1": {"inputs": {}, "_meta": {"title": "긍정프롬프트"}},
        }, None

    submitted = []

    async def monitor(_item, workflow, **_kwargs):
        prompt = workflow["1"]["inputs"]["value"]
        submitted.append(prompt)
        if "b.png" in prompt:
            raise RuntimeError("b 얼굴 검출 실패")
        return "prompt-c", {"_comfy_port": 8188}

    async def fetch_history(prompt_id, port):
        assert prompt_id == "prompt-c"
        assert port == 8188
        return {
            prompt_id: {
                "outputs": {
                    "preview": {
                        "images": [
                            {"filename": "crop.png", "subfolder": "", "type": "temp"}
                        ]
                    }
                }
            }
        }

    async def fetch_image(filename, subfolder, image_type, port):
        assert (filename, subfolder, image_type, port) == ("crop.png", "", "temp", 8188)
        return _png_bytes((10, 20, 30))

    events = []

    async def notify(event_type, data):
        events.append((event_type, data))

    manager.convert_workflow_via_endpoint = convert_workflow
    manager._monitor_training_ws = monitor
    manager.fetch_real_history = fetch_history
    manager.fetch_real_image = fetch_image
    manager.notify_frontend = notify
    item = QueueItem(
        id="dialogue-crop",
        type="instance_lora_face_extract",
        label="dialogue crop",
        params={
            "operation": "bot_dialogue_face_crop",
            "bot_name": "sample-bot",
            "char_names": ["alice"],
            "face_crop_top": 1.4,
            "face_crop_bottom": 1.1,
        },
    )

    result = await manager._handle_instance_lora_face_extract(item)

    assert result["success"] is True
    assert result["warning"] is True
    assert result["success_count"] == 2
    assert result["skipped_count"] == 0
    assert result["overwritten_count"] == 1
    assert result["failed_count"] == 1
    assert len(submitted) == 3
    assert all("[FACE_CROP_TOP]\n1.4" in prompt for prompt in submitted)
    assert all("[FACE_CROP_BOTTOM]\n1.1" in prompt for prompt in submitted)
    assert (crop_dir / "a_face.png").read_bytes() == _png_bytes((10, 20, 30))
    assert (crop_dir / "b_face.png").read_bytes() == failed_existing_bytes
    assert (crop_dir / "c_face.png").read_bytes() == _png_bytes((10, 20, 30))
    assert not (comfy_input / "soya_dialogue_face_crop" / item.id).exists()
    complete = [
        data
        for event_type, data in events
        if event_type == "bot_dialogue_face_crop_progress" and data.get("phase") == "complete"
    ]
    assert len(complete) == 1
    assert complete[0]["failed_count"] == 1


@pytest.mark.asyncio
async def test_dialogue_face_crop_all_existing_are_reprocessed_and_overwritten(tmp_path, monkeypatch):
    bot_root = tmp_path / "bot"
    char_dir = bot_root / "sample-bot" / "alice"
    crop_dir = char_dir / "FACE CROP"
    comfy_input = tmp_path / "comfy-input"
    crop_dir.mkdir(parents=True)
    comfy_input.mkdir()
    (char_dir / "alice.png").write_bytes(_png_bytes())
    (crop_dir / "alice_face.png").write_bytes(_png_bytes((1, 2, 3)))
    workflow_path = tmp_path / "face-workflow.json"
    workflow_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(bot_mode_module, "BOT_DIR", str(bot_root))

    manager = QueueManager()
    manager.get_config = lambda: {
        "comfy_input_dir": str(comfy_input),
        "face_extract_workflow_source_path": str(workflow_path),
    }

    async def convert_workflow(_raw, task_key):
        assert task_key == "face_extract"
        return {"1": {"inputs": {}, "_meta": {"title": "긍정프롬프트"}}}, None

    async def monitor(_item, _workflow, **_kwargs):
        return "prompt-existing", {"_comfy_port": 8188}

    async def fetch_history(prompt_id, port):
        assert (prompt_id, port) == ("prompt-existing", 8188)
        return {
            prompt_id: {
                "outputs": {
                    "preview": {
                        "images": [
                            {"filename": "replacement.png", "subfolder": "", "type": "temp"}
                        ]
                    }
                }
            }
        }

    replacement = _png_bytes((90, 80, 70))

    async def fetch_image(filename, subfolder, image_type, port):
        assert (filename, subfolder, image_type, port) == (
            "replacement.png", "", "temp", 8188
        )
        return replacement

    manager.convert_workflow_via_endpoint = convert_workflow
    manager._monitor_training_ws = monitor
    manager.fetch_real_history = fetch_history
    manager.fetch_real_image = fetch_image
    item = QueueItem(
        id="all-existing",
        type="instance_lora_face_extract",
        label="all existing",
        params={
            "operation": "bot_dialogue_face_crop",
            "bot_name": "sample-bot",
            "char_names": ["alice"],
        },
    )

    result = await manager._handle_instance_lora_face_extract(item)

    assert result["success"] is True
    assert result["success_count"] == 1
    assert result["skipped_count"] == 0
    assert result["overwritten_count"] == 1
    assert result["failed_count"] == 0
    assert (crop_dir / "alice_face.png").read_bytes() == replacement


def test_postprocess_prefers_saved_face_crop_without_running_onnx(monkeypatch):
    monkeypatch.setattr(
        postprocess,
        "match_face_image_filename",
        lambda *_args, **_kwargs: ("alice_happy.png", "exact", 1.0),
    )
    monkeypatch.setattr(
        postprocess,
        "load_saved_face_crop_bytes",
        lambda *_args, **_kwargs: _png_bytes((220, 30, 40)),
    )

    def original_should_not_load(*_args, **_kwargs):
        raise AssertionError("저장 FACE CROP이 있으면 원본/ONNX 경로를 사용하면 안 됩니다.")

    monkeypatch.setattr(postprocess, "load_face_image_bytes", original_should_not_load)

    result = postprocess._prepare_face_images(
        [{"speaker": "alice", "emotion": "#happy", "text": "hello"}],
        {
            "face_enabled": True,
            "strip_emotion": True,
            "face_crop_top": 1.8,
            "face_crop_bottom": 1.0,
        },
        "sample-bot",
        96,
    )

    assert "alice" in result
    assert result["alice"].getpixel((0, 0)) == (220, 30, 40)
    assert result["alice"].info["postprocess_face_center"] == (0.5, 0.5)


def test_frontend_places_face_crop_folder_last_and_has_no_folder_entry_button():
    frontend = (
        Path(__file__).resolve().parents[1] / "frontend" / "index.html"
    ).read_text(encoding="utf-8")
    render_source = frontend.split(
        "function renderBotCharDetailGrid", 1
    )[1].split("async function loadBotCharDetailImages", 1)[0]

    assert "grid.appendChild(card);" in render_source
    assert "grid.appendChild(folderCard);" in render_source
    assert render_source.index("grid.appendChild(card);") < render_source.index(
        "grid.appendChild(folderCard);"
    )
    assert "folderCard.onclick = () => openBotFaceCropFolder(charName);" in render_source
    assert "btn-face-crop-folder" not in frontend
    assert 'id="btn-dialogue-face-crop"' in frontend
    assert 'id="dialogue-face-crop-top"' in frontend
    assert 'id="dialogue-face-crop-bottom"' in frontend
    assert "face_crop_top: cropTop" in frontend
    assert "face_crop_bottom: cropBottom" in frontend
    assert "기존 <b>FACE CROP</b> 파일은 새 결과로 덮어쓰고" in frontend
