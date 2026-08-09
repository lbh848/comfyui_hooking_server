import asyncio
import base64
import importlib
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import workflow_profiles
from modes.asset_mode import AssetMode
from queue_manager import QueueItem, QueueManager


asset_mode_module = importlib.import_module("modes.asset_mode")


def _configured_mode() -> AssetMode:
    mode = AssetMode()
    mode._tags = {
        "quality": ["ilxl quality"],
        "composition": ["centered"],
        "appearances": {"look": ["blue hair"]},
        "outfits": {"uniform": ["uniform"]},
        "expressions": {"smile": ["smile"]},
        "negative": ["ilxl bad"],
        "character_negative": ["bad anatomy"],
        "artist_presets": {
            "ilxl_artist": ["ilxl artist"],
            "anima_artist": ["anima artist"],
        },
        "anima_quality": ["anima quality"],
        "anima_negative": ["anima bad"],
    }
    return mode


@pytest.mark.parametrize(
    ("profile", "has_anima", "has_ilxl", "has_sdxl_separator"),
    [
        ("ilxl", False, True, False),
        ("anima_ilxl", True, True, True),
        ("anima_only", True, False, False),
    ],
)
def test_asset_prompt_builder_covers_all_three_workflows(
    profile,
    has_anima,
    has_ilxl,
    has_sdxl_separator,
):
    mode = _configured_mode()

    positive, negative = mode.build_prompts(
        appearance="look",
        outfit="uniform",
        expression="smile",
        artist_preset="ilxl_artist",
        anima_artist_preset="anima_artist",
        asset_workflow_type=profile,
        pose_enabled=True,
        pose_data={"people": []},
        face_id_enabled=True,
    )

    assert ("[ANIMA]" in positive) is has_anima
    assert ("ilxl quality" in positive) is has_ilxl
    assert ("[SDXL]" in positive) is has_sdxl_separator
    assert ("[SDXL]" in negative) is has_sdxl_separator
    if profile == "anima_only":
        assert "[FACE_ID_ACTIVATE]\nfalse" in positive
        assert "[POSE_ACTIVATE]\nfalse" in positive
        assert "ilxl bad" not in negative


def test_ilxl_builder_disables_anima_only_runtime_options():
    mode = _configured_mode()

    positive, _negative = mode.build_prompts(
        asset_workflow_type="ilxl",
        face_lora_activate=True,
        anima_fd_activate=True,
        anima_hd_activate=True,
        anima_ed_activate=True,
        seed=1234,
    )

    assert "[FACE_LORA_ACTIVATE]\nfalse" in positive
    assert "[ANIMA_FD_ACTIVATE]\nfalse" in positive
    assert "[ANIMA_HD_ACTIVATE]\nfalse" in positive
    assert "[ANIMA_ED_ACTIVATE]\nfalse" in positive
    assert "[SEED]\n-1" in positive


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("profile", "path_attr"),
    [
        ("ilxl", "workflow_source_path"),
        ("anima_ilxl", "anima_workflow_source_path"),
        ("anima_only", "anima_only_workflow_source_path"),
    ],
)
async def test_asset_generation_selects_matching_workflow_and_storage(
    monkeypatch,
    tmp_path,
    profile,
    path_attr,
):
    monkeypatch.setattr(asset_mode_module, "ASSET_DIR", str(tmp_path / "assets"))
    mode = _configured_mode()
    mode.workflow_source_path = str(tmp_path / "ilxl.json")
    mode.anima_workflow_source_path = str(tmp_path / "anima_ilxl.json")
    mode.anima_only_workflow_source_path = str(tmp_path / "anima_only.json")
    expected_path = getattr(mode, path_attr)
    selected_paths = []
    png_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )

    async def update_workflow():
        selected_paths.append(mode.workflow_source_path)
        mode._asset_api_workflow = {}
        return True

    async def submit_workflow(_workflow, progress_callback=None):
        return png_bytes, None

    monkeypatch.setattr(mode, "update_asset_workflow", update_workflow)
    monkeypatch.setattr(mode, "_save_cached_api", lambda _workflow: None)
    monkeypatch.setattr(mode, "_log", lambda *_args, **_kwargs: None)
    mode.build_prompt_with_workflow_func = lambda _workflow, _positive, _negative: {}
    mode.submit_workflow_func = submit_workflow

    result = await mode.generate(
        character="alice",
        appearance="look",
        outfit="uniform",
        expression="smile",
        positive_prompt="ready",
        negative_prompt="",
        asset_workflow_type=profile,
    )

    assert result["success"] is True
    assert selected_paths == [expected_path]
    assert mode.workflow_source_path == str(tmp_path / "ilxl.json")
    assert (
        tmp_path / "assets" / "alice" / "uniform" / "smile" / result["filename"]
    ).is_file()


@pytest.mark.asyncio
@pytest.mark.parametrize("profile", workflow_profiles.ASSET_WORKFLOW_TYPES)
async def test_queue_asset_entrypoint_preserves_selected_profile(profile):
    manager = QueueManager()
    captured = {}

    class FakeAssetMode:
        async def generate(self, **kwargs):
            captured.update(kwargs)
            return {"success": True, "filename": "result.webp"}

    manager.asset_mode = FakeAssetMode()
    manager.get_config = lambda: {"comfy_input_dir": ""}
    item = QueueItem(
        id=f"asset-{profile}",
        type="asset_generation",
        label=profile,
        params={
            "body": {
                "character": "alice",
                "appearance": "look",
                "outfit": "uniform",
                "expression": "smile",
                "asset_workflow_type": profile,
            }
        },
    )

    result = await manager._handle_asset_generation(item)

    assert result["success"] is True
    assert captured["asset_workflow_type"] == profile


@pytest.mark.asyncio
async def test_queue_asset_entrypoint_uses_configured_profile_when_body_omits_it():
    manager = QueueManager()
    captured = {}

    class FakeAssetMode:
        async def generate(self, **kwargs):
            captured.update(kwargs)
            return {"success": True, "filename": "result.webp"}

    manager.asset_mode = FakeAssetMode()
    manager.get_config = lambda: {
        "comfy_input_dir": "",
        "asset_workflow_type": "anima_only",
    }
    item = QueueItem(
        id="asset-config-default",
        type="asset_generation",
        label="config default",
        params={"body": {"character": "alice"}},
    )

    result = await manager._handle_asset_generation(item)

    assert result["success"] is True
    assert captured["asset_workflow_type"] == "anima_only"


@pytest.mark.asyncio
async def test_asset_execution_forks_do_not_serialize_local_and_modal_generation():
    mode = _configured_mode()
    local_worker = mode.fork_for_execution()
    modal_worker = mode.fork_for_execution()
    both_started = asyncio.Event()
    release = asyncio.Event()
    active = 0
    peak = 0

    async def fake_generate_internal(*_args):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        if active == 2:
            both_started.set()
        await release.wait()
        active -= 1
        return {"success": True}

    local_worker._generate_internal = fake_generate_internal
    modal_worker._generate_internal = fake_generate_internal
    local_task = asyncio.create_task(local_worker.generate(character="local"))
    modal_task = asyncio.create_task(modal_worker.generate(character="modal"))

    try:
        await asyncio.wait_for(both_started.wait(), timeout=1)
        release.set()
        results = await asyncio.wait_for(
            asyncio.gather(local_task, modal_task),
            timeout=1,
        )
        assert peak == 2
        assert all(result["success"] for result in results)
    finally:
        release.set()
        await asyncio.gather(local_task, modal_task, return_exceptions=True)


def test_frontend_covers_single_batch_bulk_and_automatch_asset_lines():
    source = (Path(__file__).resolve().parents[1] / "frontend" / "index.html").read_text(
        encoding="utf-8"
    )

    assert '<option value="ilxl">ILXL 에셋 생성 워크플로우</option>' in source
    assert '<option value="anima_ilxl">ANIMA+ILXL 에셋 생성 워크플로우</option>' in source
    assert '<option value="anima_only">ONLY ANIMA 에셋 생성 워크플로우</option>' in source
    assert 'id="setting-anima-only-asset-workflow-source-path"' in source
    assert "function buildAssetPromptFromUI()" in source
    assert "async function startBatchGeneration()" in source
    assert "에셋 일괄 생성 설정" in source
    assert "storage_group: 'automatch_defaults'" in source
    assert "function buildBatchSlotPromptData(slot)" in source
    assert "asset_workflow_type: capabilities.type" in source
    assert "asset_workflow_type: assetWorkflowType" in source
    assert "asset_workflow_type: normalizeAssetWorkflowType(currentConfig?.asset_workflow_type)" in source
    assert 'data-workflow-availability="ilxl"' in source
    assert 'data-workflow-availability="anima"' in source
    assert 'data-workflow-availability="ipadapter"' in source
    assert '<section class="batch-bulk-section workflow-mode-control" data-workflow-availability="ipadapter">' in source
    assert 'data-at-availability="pose"' in source
    assert 'data-at-availability="ilxl"' in source
    assert "function filterAssetLorasForCapabilities(items" in source
    assert "assetCapabilities.ilxl ? '' : 'disabled'" in source
    assert "getAssetWorkflowCapabilities().anima ? '' : 'disabled'" in source
