from pathlib import Path
import importlib
import json
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from queue_manager import QueueItem, QueueManager


ROOT = Path(__file__).resolve().parents[1]
FRONTEND = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
QUEUE_SOURCE = (ROOT / "queue_manager.py").read_text(encoding="utf-8")
SERVER_SOURCE = (ROOT / "server.py").read_text(encoding="utf-8")
BOT_MODE_SOURCE = (ROOT / "modes" / "bot_mode.py").read_text(encoding="utf-8")


def _function_source(source: str, name: str, next_name: str) -> str:
    return source.split(f"function {name}", 1)[1].split(f"function {next_name}", 1)[0]


def test_one_click_parent_group_contains_the_existing_work_groups_and_dialogue_crop():
    parent = FRONTEND.split('id="bot-one-click-workflow-group"', 1)[1].split(
        'id="btn-bot-auto-lora"', 1
    )[0]

    assert 'id="btn-bot-one-click-settings"' in parent
    for expected in (
        'id="btn-bot-analyze-all"',
        'id="btn-bot-negative-all"',
        'id="btn-data-patch"',
        'id="btn-program-embedding"',
        'id="btn-dialogue-face-crop"',
        'id="btn-bot-util-analyze"',
        'id="btn-bot-llm-batch"',
    ):
        assert expected in parent


def test_one_click_modal_is_a_four_step_wizard_with_stage_toggles():
    modal = _function_source(
        FRONTEND, "openBotOneClickSettings()", "botOneClickSelectCharacters(checked)"
    )

    assert "BOT_ONE_CLICK_WIZARD_STEPS.map" in modal
    assert modal.count('class="bot-one-click-wizard-page"') == 4
    assert 'class="bot-one-click-stage-toggle"' in modal
    assert 'id="bot-one-click-stage-list"' in modal
    assert 'id="bot-one-click-char-list"' in modal
    assert 'id="bot-one-click-negative-preset"' in modal
    assert 'id="bot-one-click-negative-tags"' in modal
    assert 'id="bot-one-click-crop-top"' in modal
    assert 'id="bot-one-click-crop-bottom"' in modal
    assert 'id="bot-one-click-confidence"' in modal
    assert 'id="bot-one-click-overwrite"' in modal
    assert 'id="bot-one-click-review"' in modal
    assert 'id="bot-one-click-back-btn"' in modal
    assert 'id="bot-one-click-next-btn"' in modal
    assert 'id="bot-one-click-start-btn"' in modal


def test_one_click_wizard_validates_only_settings_needed_by_enabled_stages():
    validation = _function_source(
        FRONTEND,
        "_botOneClickValidateWizardStep(step)",
        "_botOneClickRenderReview()",
    )
    visibility = _function_source(
        FRONTEND,
        "_botOneClickUpdateSettingVisibility()",
        "_botOneClickWizardError(message)",
    )

    for source in (validation, visibility):
        assert "['negative', 'face_batch']" in source
        assert "['data_patch', 'embedding', 'dialogue_face_crop']" in source
    assert "needsNegative && !negativeTags" in validation
    assert "needsCrop && (!Number.isFinite(cropTop)" in validation
    assert "enabledStages.embedding && (!Number.isFinite(confidence)" in validation
    assert "실행할 기능을 하나 이상 켜세요." in validation


def test_one_click_disabled_stages_are_snapshotted_and_skipped():
    start = _function_source(
        FRONTEND, "startBotOneClick()", "_botOneClickRunStage(key, task)"
    )
    pipeline = _function_source(
        FRONTEND, "_runBotOneClickPipeline(run)", "_botOneClickTrackTagBatch"
    )
    progress = _function_source(
        FRONTEND, "_botOneClickRenderProgress()", "startBotOneClick()"
    )

    assert "enabledStages," in start
    assert "status: enabledStages[key] ? 'pending' : 'skipped'" in start
    assert "if (!run.enabledStages?.[key]) return;" in pipeline
    assert "runSelectedStage('tag_analysis'" in pipeline
    assert "runSelectedStage('llm_refine'" in pipeline
    assert "skipped: '–'" in progress
    assert "건너뜀 ${skipped}단계" in progress


def test_one_click_pipeline_order_and_transient_patch_settings_are_explicit():
    pipeline = _function_source(
        FRONTEND, "_runBotOneClickPipeline(run)", "_botOneClickTrackTagBatch"
    )
    ordered_stages = (
        "tag_analysis",
        "negative",
        "data_patch",
        "embedding",
        "dialogue_face_crop",
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

    dialogue_crop = _function_source(
        FRONTEND,
        "_runBotOneClickDialogueFaceCrop(run)",
        "_runBotOneClickFaceBatch(run)",
    )
    assert "/api/bot_mode/dialogue_face_crop" in dialogue_crop
    assert "face_crop_top: run.cropTop" in dialogue_crop
    assert "face_crop_bottom: run.cropBottom" in dialogue_crop

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


def test_one_click_finished_result_can_be_reopened_and_starts_new_setup_explicitly():
    open_settings = _function_source(
        FRONTEND, "openBotOneClickSettings()", "botOneClickSelectCharacters(checked)"
    )
    start = _function_source(
        FRONTEND, "startBotOneClick()", "_botOneClickRunStage(key, task)"
    )
    progress = _function_source(
        FRONTEND, "_botOneClickRenderProgress()", "startBotOneClick()"
    )

    assert "_botOneClickRun?.botName === botCurrentBot" in open_settings
    assert "button.textContent = '원클릭 결과 보기'" in start
    assert 'onclick="openNewBotOneClickSettings()"' in progress
    assert "function openNewBotOneClickSettings()" in FRONTEND


def test_one_click_progress_modal_exposes_safe_stop_and_marks_remaining_stages_cancelled():
    progress = _function_source(
        FRONTEND, "_botOneClickRenderProgress()", "startBotOneClick()"
    )
    stop = _function_source(
        FRONTEND, "requestBotOneClickStop()", "_botOneClickStageEntry(key)"
    )
    pipeline = _function_source(
        FRONTEND, "_runBotOneClickPipeline(run)", "_botOneClickTrackTagBatch"
    )

    assert 'id="bot-one-click-stop-btn"' in progress
    assert 'onclick="requestBotOneClickStop()"' in progress
    assert "'안전 중단'" in progress
    assert "window.confirm(" in stop
    assert "run.stopRequested = true" in stop
    assert "await _botOneClickCancelPendingWork(run)" in stop
    assert "stage.status = 'cancelled'" in pipeline
    assert "안전 중단으로 실행하지 않음" in pipeline


def test_one_click_safe_stop_tracks_only_its_queue_items_with_a_run_token():
    cancel = _function_source(
        FRONTEND,
        "_botOneClickCancelPendingWork(run)",
        "requestBotOneClickStop()",
    )
    start = _function_source(
        FRONTEND, "startBotOneClick()", "_botOneClickRunStage(run, key, task)"
    )
    one_click_source = _function_source(
        FRONTEND, "_runBotOneClickPipeline(run)", "openDialogueFaceCropModal()"
    )

    assert "runId:" in start
    assert "queuedItems: []" in start
    assert "tagBatches: []" in start
    assert "'/api/queue/cancel_one_click'" in cancel
    assert "one_click_run_id: run.runId" in cancel
    assert "'/api/queue/cancel_batch'" in FRONTEND
    assert "fetch('/api/queue/cancel'" in FRONTEND
    assert one_click_source.count("one_click_run_id: run.runId") >= 5
    assert 'app.router.add_post("/api/queue/cancel_one_click"' in SERVER_SOURCE
    assert 'params["one_click_run_id"] = one_click_run_id' in BOT_MODE_SOURCE


def test_one_click_warning_result_exposes_stage_character_file_and_reason():
    progress = _function_source(
        FRONTEND, "_botOneClickRenderProgress()", "startBotOneClick()"
    )
    warning_summary = _function_source(
        FRONTEND, "_botOneClickWarningSummary(run)", "_botOneClickStageEntry(key)"
    )
    dialogue_crop = _function_source(
        FRONTEND,
        "_runBotOneClickDialogueFaceCrop(run)",
        "_runBotOneClickFaceBatch(run)",
    )
    llm_refine = _function_source(
        FRONTEND, "_runBotOneClickLlmRefine(run)", "openDialogueFaceCropModal()"
    )

    assert "stage.issues" in progress
    assert "issue.character" in progress
    assert "issue.filename" in progress
    assert "issue.reason" in progress
    assert "문제 캐릭터:" in warning_summary
    assert "result.character_warnings" in dialogue_crop
    assert "result.failed" in dialogue_crop
    assert "result.failed.map" in llm_refine


def test_one_click_tag_and_negative_failures_keep_character_metadata():
    assert '"character": img.get("character", "")' in QUEUE_SOURCE
    assert '"char_name": rep["character"]' in BOT_MODE_SOURCE
    assert '"failed": failed' in BOT_MODE_SOURCE


@pytest.mark.asyncio
async def test_utility_negative_failure_response_names_character_and_file(
    tmp_path, monkeypatch
):
    bot_mode_module = importlib.import_module("modes.bot_mode")
    manager = bot_mode_module.BotMode()
    monkeypatch.setattr(bot_mode_module, "BOT_DIR", str(tmp_path / "bot"))
    manager._get_utility_image_paths = lambda _bot, _character: [{
        "character": "alice",
        "filename": "_face_image.webp",
    }]

    class Request:
        async def json(self):
            return {
                "bot": "sample-bot",
                "characters": ["alice"],
                "negative_tags": "bad anatomy",
            }

    response = await manager.handle_batch_set_negative_utility(Request())
    payload = json.loads(response.text)

    assert payload["success_count"] == 0
    assert payload["fail_count"] == 1
    assert len(payload["failed"]) == 1
    failure = payload["failed"][0]
    assert failure["char_name"] == "alice"
    assert failure["filename"] == "_face_image.webp"
    assert failure["error"]


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
async def test_one_click_safe_stop_cancels_matching_pending_and_rejects_late_items():
    manager = QueueManager()
    manager._paused = True

    async def no_llm_workers():
        return None

    async def no_notify():
        return None

    manager._ensure_llm_workers = no_llm_workers
    manager._notify_queue_updated = no_notify
    run_id = "one-click-safe-stop-test"
    matching_pending = await manager.add_item(
        "data_patch_utility",
        "matching pending",
        {"one_click_run_id": run_id},
    )
    matching_processing = await manager.add_item(
        "data_patch_utility",
        "matching processing",
        {"one_click_run_id": run_id},
    )
    unrelated = await manager.add_item(
        "data_patch_utility",
        "unrelated",
        {"one_click_run_id": "another-run"},
    )
    matching_processing.status = "processing"

    cancelled = await manager.cancel_one_click_run(run_id)
    late = await manager.add_item(
        "data_patch_utility",
        "late matching item",
        {"one_click_run_id": run_id},
    )

    assert cancelled == 1
    assert matching_pending.status == "cancelled"
    assert matching_processing.status == "processing"
    assert unrelated.status == "pending"
    assert late.status == "cancelled"


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
