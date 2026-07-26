import asyncio
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import server
import workflow_profiles
from modes import illustration_context_pipeline as pipeline
from queue_manager import QueueManager


class _JsonRequest:
    method = "POST"

    def __init__(self, body):
        self._body = body

    async def json(self):
        return copy.deepcopy(self._body)


@pytest.mark.parametrize(
    ("profile", "provider", "prompt_format", "local_profile", "restore_family"),
    [
        ("v1", "comfy", "v1", "v1", "v1"),
        ("v3", "comfy", "v3", "v3", "v3"),
        ("v3_anima", "comfy", "v3", "v3_anima", "v3"),
        ("chansub", "chansub", "chansub", None, "chansub"),
        ("chansub_v3_anima", "hybrid", "v3", "v3_anima", "v3+chansub"),
    ],
)
def test_all_illustration_profiles_normalize_end_to_end(
    profile,
    provider,
    prompt_format,
    local_profile,
    restore_family,
):
    source_paths = {
        "v1": r"E:\workflows\배포_삽화_v1_1.json",
        "v3": r"E:\workflows\배포_삽화_v3_4.json",
        "v3_anima": r"E:\workflows\배포_삽화(ONLY_ANIMA)_v3_4.json",
    }
    config = {
        "illustration_workflow_type": profile,
        "illustration_provider": "wrong-manual-value",
        "illustration_workflow_source_paths": copy.deepcopy(source_paths),
        "illustration_context_toggles": {"prompt_format": "wrong-manual-value"},
        "chansub_workflow_type": "sdxl",
        "asset_workflow_type": "regular",
    }

    workflow_profiles.normalize_workflow_config(config)

    assert config["illustration_provider"] == provider
    assert config["illustration_context_toggles"]["prompt_format"] == prompt_format
    assert workflow_profiles.illustration_local_profile(profile) == local_profile
    assert workflow_profiles.restore_family(profile) == restore_family
    assert config["asset_workflow_type"] == "ilxl"
    if local_profile:
        assert workflow_profiles.active_illustration_source_path(config) == source_paths[local_profile]


def test_restore_prompt_compatibility_matrix_covers_v1_v3_and_chansub():
    files = list(workflow_profiles.RESTORE_PROMPT_COMPATIBILITY)

    assert workflow_profiles.compatible_restore_prompt_files("v1", files) == [
        "restore_workflow_prompt_nikke_style_v2.py",
        "restore_workflow_prompt_terminater_style.py",
        "restore_workflow_prompt_nikke_style_v3.py",
    ]
    assert workflow_profiles.compatible_restore_prompt_files("v3_anima", files) == [
        "restore_workflow_prompt_llm_solo.py",
        "restore_workflow_prompt_nikke_style_v3.py",
    ]
    assert workflow_profiles.compatible_restore_prompt_files(
        "chansub_v3_anima", files
    ) == [
        "restore_workflow_prompt_llm_solo.py",
        "restore_workflow_prompt_nikke_style_v3.py",
    ]
    assert not workflow_profiles.is_restore_prompt_compatible(
        "restore_workflow_prompt_nikke_style_v2.py", "v3"
    )


@pytest.mark.asyncio
async def test_config_api_rejects_incompatible_restore_profile_without_writing(
    monkeypatch,
):
    config = copy.deepcopy(server.app_config)
    config["illustration_workflow_type"] = "v1"
    config["restore_prompt_file"] = "restore_workflow_prompt_nikke_style_v2.py"
    saved = []
    monkeypatch.setattr(server, "app_config", config)
    monkeypatch.setattr(server, "save_config", lambda value: saved.append(value))

    response = await server.handle_api_config(
        _JsonRequest({"illustration_workflow_type": "v3"})
    )

    assert response.status == 400
    assert "호환되지 않습니다" in json.loads(response.text)["error"]
    assert saved == []


@pytest.mark.asyncio
async def test_config_api_saves_hybrid_and_asset_profiles_as_derived_contract(
    monkeypatch,
):
    config = copy.deepcopy(server.app_config)
    saved = []
    monkeypatch.setattr(server, "app_config", config)
    monkeypatch.setattr(server, "save_config", lambda value: saved.append(copy.deepcopy(value)))
    monkeypatch.setattr(server.llm_service, "update_config", lambda _value: None)
    monkeypatch.setattr(server.embedding_service, "update_config", lambda _value: None)
    monkeypatch.setattr(server.asset_mode, "workflow_type", server.asset_mode.workflow_type)

    paths = {
        "v1": r"E:\workflows\배포_삽화_v1_1.json",
        "v3": r"E:\workflows\배포_삽화_v3_4.json",
        "v3_anima": r"E:\workflows\배포_삽화(ONLY_ANIMA)_v3_4.json",
    }
    response = await server.handle_api_config(
        _JsonRequest(
            {
                "illustration_workflow_type": "chansub_v3_anima",
                "illustration_workflow_source_paths": paths,
                "restore_prompt_file": "restore_workflow_prompt_nikke_style_v3.py",
                "chansub_workflow_type": "sdxl",
                "asset_workflow_type": "anima_only",
            }
        )
    )
    result = json.loads(response.text)

    assert response.status == 200
    assert result["success"] is True
    assert saved[-1]["illustration_provider"] == "hybrid"
    assert saved[-1]["illustration_context_toggles"]["prompt_format"] == "v3"
    assert saved[-1]["chansub_workflow_type"] == "anima"
    assert saved[-1]["illustration_workflow_source_paths"] == paths
    assert saved[-1]["asset_workflow_type"] == "anima_only"


@pytest.mark.asyncio
async def test_automatic_restore_uses_compatible_illustration_builder(monkeypatch):
    captured = {}
    restore_ids = []

    async def fake_process_prompt(prompt_id, prompt_data, raw_body, **_kwargs):
        restore_ids.append(prompt_id)
        captured["positive"] = server.extract_prompts_by_title(
            prompt_data, "긍정프롬프트"
        )
        captured["raw_body"] = raw_body
        server.prompts[prompt_id]["image_bytes"] = b"restore-image"

    async def fake_notify(event, data):
        captured["notification"] = (event, data)

    monkeypatch.setitem(server.app_config, "restore_mode_enabled", True)
    monkeypatch.setitem(
        server.app_config,
        "restore_prompt_file",
        "restore_workflow_prompt_nikke_style_v3.py",
    )
    monkeypatch.setitem(
        server.app_config, "illustration_workflow_type", "chansub_v3_anima"
    )
    monkeypatch.setitem(server.app_config, "illustration_provider", "hybrid")
    monkeypatch.setitem(server.app_config, "bot_selected", "RESTORE_TEST_BOT")
    monkeypatch.setattr(server, "process_prompt", fake_process_prompt)
    monkeypatch.setattr(server, "notify_frontend", fake_notify)

    try:
        await server._do_restore_workflow()

        assert "[SETUP]" in captured["positive"]
        assert "[CHAR]" in captured["positive"]
        assert captured["raw_body"] == {
            "illustration_provider": "comfy",
            "illustration_gen_method": "자동 복원",
        }
        assert captured["notification"][0] == "restore_image_saved"
    finally:
        for prompt_id in restore_ids:
            server.prompts.pop(prompt_id, None)


@pytest.mark.asyncio
async def test_hybrid_context_line_distributes_scenes_to_separate_builders(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(pipeline, "SESSION_DIR", str(tmp_path / "sessions"))
    session_id = "risu_" + ("a" * 64)
    original_prompt_id = "hybrid-e2e-parent"
    pipeline.create_session(session_id, "")
    server.prompts[original_prompt_id] = {
        "status": "running",
        "prompt": {},
        "outputs": {},
        "filename": None,
        "save_node_id": "9",
        "image_bytes": None,
    }
    descriptors = [
        {
            "kind": "scene",
            "slot": slot,
            "raw_positive": f"positive {slot}",
            "raw_negative": f"negative {slot}",
        }
        for slot in range(4)
    ]
    enqueued = []
    child_prompt_ids = []

    async def fake_add_item(item_type, label, params, priority=10, **kwargs):
        assert item_type == "illustration"
        child_id = params["prompt_id"]
        child_prompt_ids.append(child_id)
        snapshot = server.prompts[child_id]["_illustration_runtime_snapshot"]
        enqueued.append(
            (
                params["provider"],
                params["raw_body"]["illustration_prompt_format"],
                snapshot["provider"],
                snapshot["illustration_workflow_type"],
            )
        )
        future = asyncio.get_running_loop().create_future()
        item = SimpleNamespace(status="completed", completion_future=future)
        server.prompts[child_id]["image_bytes"] = f"image-{len(enqueued)}".encode()
        future.set_result({"success": True})
        return item

    async def ignore_progress(*args, **kwargs):
        return None

    async def fake_complete_prompt(prompt_id, save_node_id, filename):
        server.prompts[prompt_id]["status"] = "completed"

    monkeypatch.setitem(server.app_config, "bot_selected", "HYBRID_TEST_BOT")
    monkeypatch.setitem(
        server.app_config, "illustration_workflow_type", "chansub_v3_anima"
    )
    monkeypatch.setitem(server.app_config, "illustration_provider", "hybrid")
    monkeypatch.setitem(server.app_config, "chansub_workflow_type", "sdxl")
    monkeypatch.setitem(
        server.app_config,
        "illustration_context_toggles",
        {"prompt_format": "v1"},
    )
    monkeypatch.setattr(server, "_load_word_rules_snapshot", lambda _bot: [])
    monkeypatch.setattr(server.queue_manager, "add_item", fake_add_item)
    monkeypatch.setattr(server.queue_manager, "_notify_progress", ignore_progress)
    monkeypatch.setattr(server, "set_prompt_by_title", lambda *args, **kwargs: True)
    monkeypatch.setattr(server, "complete_prompt_from_reschedule", fake_complete_prompt)

    parent_item = SimpleNamespace(
        params={
            "prompt_id": original_prompt_id,
            "payload": {
                "protocol": "prompt_batch_v1",
                "session_id": session_id,
                "items": descriptors,
            },
            "prompt_data": {},
            "raw_body": {},
        }
    )

    try:
        result = await server.process_illustration_context_queue_item(parent_item)

        assert enqueued == [
            ("comfy", "v3", "comfy", "chansub_v3_anima"),
            ("chansub", "chansub", "chansub", "chansub_v3_anima"),
            ("comfy", "v3", "comfy", "chansub_v3_anima"),
            ("chansub", "chansub", "chansub", "chansub_v3_anima"),
        ]
        assert result["count"] == 4
        assert pipeline.get_session(session_id)["images"] == [
            b"image-1",
            b"image-2",
            b"image-3",
            b"image-4",
        ]
    finally:
        pipeline._SESSIONS.pop(session_id, None)
        pipeline._LOOKUP_KEYS.pop("a" * 24, None)
        server.prompts.pop(original_prompt_id, None)
        for child_id in child_prompt_ids:
            server.prompts.pop(child_id, None)


@pytest.mark.asyncio
async def test_local_and_chansub_queue_lanes_execute_concurrently(monkeypatch):
    manager = QueueManager()
    manager.get_config = lambda: {
        "bot_selected": "HYBRID_TEST_BOT",
        "illustration_provider": "hybrid",
    }
    both_started = asyncio.Event()
    release = asyncio.Event()
    started = set()

    async def fake_execute(item):
        started.add(manager._item_execution_area(item)[0])
        if started == {"gpu", "external"}:
            both_started.set()
        await release.wait()
        return {"success": True}

    async def no_llm_workers():
        return None

    async def no_wait():
        return None

    async def no_prune(_item):
        return None

    monkeypatch.setattr(manager, "_execute_item", fake_execute)
    monkeypatch.setattr(manager, "_ensure_llm_workers", no_llm_workers)
    monkeypatch.setattr(manager, "_wait_after_illustration", no_wait)
    monkeypatch.setattr(manager, "_deferred_prune", no_prune)

    local = await manager.add_item(
        "illustration",
        "local",
        {"raw_body": {"illustration_provider": "comfy"}},
        priority=0,
    )
    remote = await manager.add_item(
        "illustration",
        "remote",
        {"raw_body": {"illustration_provider": "chansub"}},
        priority=0,
    )

    try:
        await asyncio.wait_for(both_started.wait(), timeout=1)
        assert local.status == "processing"
        assert remote.status == "processing"
        assert manager.get_status()["processing"] is True
        assert manager.get_status()["current_external"]["id"] == remote.id
        release.set()
        await asyncio.wait_for(
            asyncio.gather(local.completion_future, remote.completion_future),
            timeout=1,
        )
        assert local.status == remote.status == "completed"
    finally:
        release.set()
        if manager._external_worker_task and not manager._external_worker_task.done():
            manager._external_worker_task.cancel()
            await asyncio.gather(manager._external_worker_task, return_exceptions=True)


def test_frontend_shows_all_profiles_and_disables_manual_prompt_format():
    source = (Path(__file__).resolve().parents[1] / "frontend" / "index.html").read_text(
        encoding="utf-8"
    )

    for profile in ("v1", "v3", "v3_anima", "chansub", "chansub_v3_anima"):
        assert f'<option value="{profile}">' in source
    assert 'id="setting-illustration-workflow-type"' in source
    assert "select.disabled = true;" in source
    assert "opt.disabled = !requiredFamilies.every(family => families.includes(family));" in source
    assert "group.classList.toggle('disabled', disabled);" in source
    assert "if (!isV1) updateChansubAdvancedOptions();" in source
    assert 'data-illust-capability="ipadapter"' in source
