from pathlib import Path
import importlib
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from queue_manager import QueueItem, QueueManager


ROOT = Path(__file__).resolve().parents[1]
FRONTEND = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
QUEUE_SOURCE = (ROOT / "queue_manager.py").read_text(encoding="utf-8")
SERVER_SOURCE = (ROOT / "server.py").read_text(encoding="utf-8")


def _function_source(source: str, name: str, next_name: str) -> str:
    return source.split(f"function {name}", 1)[1].split(f"function {next_name}", 1)[0]


def test_one_click_parent_group_contains_the_six_existing_work_groups():
    parent = FRONTEND.split('id="bot-one-click-workflow-group"', 1)[1].split(
        'id="btn-bot-auto-lora"', 1
    )[0]

    assert 'id="btn-bot-one-click-settings"' in parent
    for expected in (
        'id="btn-bot-analyze-all"',
        'id="btn-bot-negative-all"',
        'id="btn-data-patch"',
        'id="btn-program-embedding"',
        'id="btn-bot-util-analyze"',
        'id="btn-bot-llm-batch"',
    ):
        assert expected in parent


def test_one_click_modal_has_only_target_negative_and_face_setup_inputs():
    modal = _function_source(
        FRONTEND, "openBotOneClickSettings()", "botOneClickSelectCharacters(checked)"
    )

    assert 'id="bot-one-click-char-list"' in modal
    assert 'id="bot-one-click-negative-preset"' in modal
    assert 'id="bot-one-click-negative-tags"' in modal
    assert 'id="bot-one-click-crop-top"' in modal
    assert 'id="bot-one-click-crop-bottom"' in modal
    assert 'id="bot-one-click-confidence"' in modal
    assert 'id="bot-one-click-overwrite"' in modal
    assert 'id="bot-one-click-start-btn"' in modal


def test_one_click_pipeline_order_and_transient_patch_settings_are_explicit():
    pipeline = _function_source(
        FRONTEND, "_runBotOneClickPipeline(run)", "_botOneClickTrackTagBatch"
    )
    ordered_stages = (
        "tag_analysis",
        "negative",
        "data_patch",
        "embedding",
        "face_batch",
        "llm_refine",
    )
    indexes = [pipeline.index(f"'{stage}'") for stage in ordered_stages]
    assert indexes == sorted(indexes)

    data_patch = _function_source(
        FRONTEND, "_runBotOneClickDataPatch(run)", "_runBotOneClickEmbedding(run)"
    )
    assert "patch_settings:" in data_patch
    assert "face_crop_top: run.cropTop" in data_patch
    assert "face_crop_bottom: run.cropBottom" in data_patch
    assert "/api/bot_mode/patch_settings" not in data_patch

    assert "patch_settings=patch_settings" in QUEUE_SOURCE
    assert "저장하지 않는 임시 패치 설정 적용" in SERVER_SOURCE


def test_one_click_completion_trackers_are_connected_to_websocket_handlers():
    data_patch_handler = _function_source(
        FRONTEND, "handleDataPatchProgress(data)", "togglePromptPreviewModal"
    )
    llm_handler = _function_source(
        FRONTEND, "_handleBotLlmFaceTagProgress(data)", "_refreshBotDataSilent()"
    )

    assert "_handleBotOneClickDataPatchProgress(data);" in data_patch_handler
    assert "_handleBotOneClickLlmProgress(data);" in llm_handler


@pytest.mark.asyncio
async def test_data_patch_queue_forwards_per_run_patch_settings():
    manager = QueueManager()
    received = {}

    async def run_utility(bot_name, char_name, patch_settings=None):
        received.update(
            bot_name=bot_name,
            char_name=char_name,
            patch_settings=patch_settings,
        )
        return {"success": True}

    async def notify_progress(_item, _data):
        return None

    manager.run_data_patch_utility = run_utility
    manager._notify_progress = notify_progress
    manager.notify_frontend = None
    settings = {
        "face_crop_top": 1.4,
        "face_crop_bottom": 1.2,
        "emb_target": "대표만",
    }
    item = QueueItem(
        id="one-click-patch",
        type="data_patch_utility",
        label="one click",
        params={
            "bot_name": "sample-bot",
            "char_name": "alice",
            "patch_settings": settings,
        },
    )

    result = await manager._handle_data_patch_utility(item)

    assert result["success"] is True
    assert received == {
        "bot_name": "sample-bot",
        "char_name": "alice",
        "patch_settings": settings,
    }


@pytest.mark.asyncio
async def test_data_patch_utility_uses_transient_crop_and_backs_up_existing_face(
    tmp_path, monkeypatch
):
    import server
    bot_mode = importlib.import_module("modes.bot_mode")

    bot_root = tmp_path / "bot"
    char_dir = bot_root / "sample-bot" / "alice"
    char_dir.mkdir(parents=True)
    face_path = char_dir / "_face_image.webp"
    prompt_path = char_dir / "_face_image_prompt.json"
    face_path.write_bytes(b"old-face")
    prompt_path.write_text('{"prompt":"old"}', encoding="utf-8")
    captured = {}

    async def load_workflow():
        return {"1": {"inputs": {}, "_meta": {"title": "긍정프롬프트"}}}, None

    async def submit_workflow(workflow, task_key):
        captured["workflow"] = workflow
        captured["task_key"] = task_key
        return b"new-face", None

    def build_prompt(_bot_name, _char_name, settings):
        captured["settings"] = dict(settings)
        return "utility prompt"

    monkeypatch.setattr(server, "BASE_DIR", str(tmp_path))
    monkeypatch.setattr(bot_mode, "BOT_DIR", str(bot_root))
    monkeypatch.setattr(
        bot_mode,
        "_load_bot_data",
        lambda: {
            "bots": [
                {
                    "name": "sample-bot",
                    "characters": [{"name": "alice", "rep_images": ["rep.webp"]}],
                }
            ]
        },
    )
    monkeypatch.setattr(
        bot_mode,
        "_load_patch_settings",
        lambda _bot_name: {
            "face_crop_top": 9.0,
            "face_crop_bottom": 9.0,
            "emb_target": "둘다",
        },
    )
    monkeypatch.setattr(bot_mode, "build_utility_prompt", build_prompt)
    monkeypatch.setattr(server.data_patcher, "_load_utility_workflow", load_workflow)
    monkeypatch.setattr(server, "submit_workflow_to_comfy", submit_workflow)

    result = await server._run_data_patch_utility(
        "sample-bot",
        "alice",
        patch_settings={
            "face_crop_top": 1.4,
            "face_crop_bottom": 1.2,
            "emb_target": "대표만",
        },
    )

    assert result["character"] == "alice"
    assert captured["settings"] == {
        "face_crop_top": 1.4,
        "face_crop_bottom": 1.2,
        "emb_target": "대표만",
    }
    assert face_path.read_bytes() == b"new-face"
    assert not prompt_path.exists()
    backups = list((tmp_path / "요구사항").glob("data_patch_backup_*"))
    assert len(backups) == 1
    backup_char = backups[0] / "sample-bot" / "alice"
    assert (backup_char / "_face_image.webp").read_bytes() == b"old-face"
    assert (backup_char / "_face_image_prompt.json").read_text(encoding="utf-8") == '{"prompt":"old"}'
