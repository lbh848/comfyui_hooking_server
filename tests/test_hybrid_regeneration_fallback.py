import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import server
from modes import illustration_context_pipeline as pipeline
from queue_manager import QueueItem, QueueManager


def test_backup_provider_mode_requires_explicit_hybrid_metadata(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(server, "WORKFLOW_BACKUP_DIR", str(tmp_path))
    (tmp_path / "hybrid_info.json").write_text(
        json.dumps({
            "provider": "chansub",
            "provider_mode": "hybrid",
            "prompt_provider": "comfy",
        }),
        encoding="utf-8",
    )
    (tmp_path / "legacy_info.json").write_text(
        json.dumps({"provider": "chansub"}),
        encoding="utf-8",
    )

    assert server._read_backup_provider_mode("hybrid", "chansub") == "hybrid"
    assert server._read_backup_prompt_provider("hybrid", "chansub") == "comfy"
    assert server._backup_uses_hybrid_regeneration("hybrid") is True
    assert server._read_backup_provider_mode("legacy", "chansub") == "chansub"
    assert server._read_backup_prompt_provider("legacy", "chansub") == "chansub"
    assert server._backup_uses_hybrid_regeneration("legacy") is False


def test_frontend_prompt_editor_prefers_prompt_provider_metadata():
    source = (
        Path(__file__).resolve().parents[1] / "frontend" / "index.html"
    ).read_text(encoding="utf-8")

    assert "(promptData.conversion_info || {}).prompt_provider" in source


@pytest.mark.parametrize(
    ("raw_body", "expected_mode"),
    [
        ({"illustration_provider": "comfy"}, "comfy"),
        (
            {
                "illustration_provider": "comfy",
                "illustration_provider_mode": "hybrid",
            },
            "hybrid",
        ),
    ],
)
@pytest.mark.asyncio
async def test_process_prompt_requires_explicit_hybrid_mode_for_bound_provider(
    monkeypatch,
    raw_body,
    expected_mode,
):
    prompt_id = f"provider-mode-{expected_mode}"
    captured = {}
    server.prompts[prompt_id] = {
        "status": "running",
        "outputs": {},
        "filename": None,
        "save_node_id": "9",
        "image_bytes": None,
        "_illustration_runtime_snapshot": {
            "bot_name": "test-bot",
            "provider": "hybrid",
            "illustration_workflow_type": "chansub_v3_anima",
            "chansub_workflow_type": "anima",
            "clamp_enabled": False,
            "clamp_value": 1.2,
            "word_rules": [],
        },
    }

    def fake_extract(_prompt, title):
        if title == "긍정프롬프트":
            return "[Positive]\nscene\n\n[ILXL]\nscene\n\n[UPSCALE]\nscene"
        return "negative"

    async def fake_generate(positive, negative, **kwargs):
        assert kwargs["provider"] == "comfy"
        return b"generated-image", {}

    async def fake_save_backup(image_bytes, *args, **kwargs):
        captured.update(kwargs)
        return "saved-backup", image_bytes

    monkeypatch.setattr(server, "extract_prompts_by_title", fake_extract)
    monkeypatch.setattr(server, "generate_image_with_prompt", fake_generate)
    monkeypatch.setattr(server, "save_backup", fake_save_backup)

    try:
        await server.process_prompt(prompt_id, {}, raw_body)
        assert captured["provider"] == "comfy"
        assert captured["provider_mode"] == expected_mode
        assert captured["prompt_provider"] == "comfy"
    finally:
        server.prompts.pop(prompt_id, None)


@pytest.mark.parametrize(
    ("source_provider", "expected_providers"),
    [
        ("chansub", ["chansub", "comfy"]),
        ("comfy", ["comfy", "chansub"]),
    ],
)
@pytest.mark.asyncio
async def test_hybrid_regeneration_uses_source_then_opposite_provider(
    monkeypatch,
    source_provider,
    expected_providers,
):
    calls = []

    async def fake_add_item(item_type, label, params, priority=10, **_kwargs):
        calls.append((item_type, label, params, priority))
        future = asyncio.get_running_loop().create_future()
        item = SimpleNamespace(
            completion_future=future,
            generated_image_bytes=b"fallback-image",
        )
        if len(calls) == 1:
            future.set_exception(RuntimeError(f"{source_provider} unavailable"))
        else:
            future.set_result({
                "success": True,
                "generation_time": 1.5,
                "backup_name": "fallback-backup",
            })
        return item

    monkeypatch.setattr(server.queue_manager, "add_item", fake_add_item)

    result, item = await server._run_regeneration_attempts(
        backup_name="source-backup",
        positive="edited positive",
        negative="edited negative",
        bot_name="test-bot",
        postprocess_settings=None,
        speak_text="",
        source_provider=source_provider,
        provider_mode="hybrid",
        prompt_provider=source_provider,
        generation_params={"width": 1024, "height": 1024},
        illustration_multi_char=None,
        illustration_visual_states={
            "Hero": {"visual_profile_id": "awakened"},
        },
        label="수정재생성",
    )

    assert [call[2]["provider"] for call in calls] == expected_providers
    assert all(call[2]["provider_mode"] == "hybrid" for call in calls)
    assert all(
        call[2]["prompt_provider"] == source_provider for call in calls
    )
    assert all(call[2]["positive"] == "edited positive" for call in calls)
    assert all(call[2]["illustration_visual_states"] == {
        "Hero": {"visual_profile_id": "awakened"},
    } for call in calls)
    assert result["provider"] == expected_providers[-1]
    assert result["provider_mode"] == "hybrid"
    assert result["fallback_used"] is True
    assert item.generated_image_bytes == b"fallback-image"


@pytest.mark.asyncio
async def test_non_hybrid_regeneration_never_tries_opposite_provider(monkeypatch):
    providers = []

    async def fake_add_item(_item_type, _label, params, priority=10, **_kwargs):
        providers.append(params["provider"])
        future = asyncio.get_running_loop().create_future()
        future.set_exception(RuntimeError("single provider failed"))
        return SimpleNamespace(completion_future=future)

    monkeypatch.setattr(server.queue_manager, "add_item", fake_add_item)

    with pytest.raises(RuntimeError, match="모두 실패"):
        await server._run_regeneration_attempts(
            backup_name="single-provider-backup",
            positive="positive",
            negative="negative",
            bot_name="test-bot",
            postprocess_settings=None,
            speak_text="",
            source_provider="comfy",
            provider_mode="comfy",
            prompt_provider="comfy",
            generation_params={},
            illustration_multi_char=None,
            label="재생성",
        )

    assert providers == ["comfy"]


@pytest.mark.asyncio
async def test_regenerate_queue_preserves_hybrid_mode_in_new_backup():
    manager = QueueManager()
    saved = {}

    async def fake_generate(
        positive,
        negative,
        progress_callback=None,
        provider="comfy",
        width=None,
        height=None,
    ):
        assert provider == "chansub"
        return b"generated-image", {}

    async def fake_save_backup(image_bytes, *args, **kwargs):
        saved.update(kwargs)
        return "new-backup", image_bytes

    manager.generate_image_with_prompt = fake_generate
    manager.save_backup = fake_save_backup
    item = QueueItem(
        id="hybrid-regenerate",
        type="regenerate",
        label="regenerate",
        params={
            "backup_name": "source-backup",
            "positive": "positive",
            "negative": "negative",
            "provider": "chansub",
            "provider_mode": "hybrid",
            "prompt_provider": "comfy",
            "generation_params": {"width": 1024, "height": 1024},
            "illustration_visual_states": {
                "Hero": {"visual_profile_id": "awakened"},
            },
        },
    )

    result = await manager._handle_regenerate(item)

    assert result["provider"] == "chansub"
    assert result["provider_mode"] == "hybrid"
    assert result["prompt_provider"] == "comfy"
    assert saved["provider"] == "chansub"
    assert saved["provider_mode"] == "hybrid"
    assert saved["prompt_provider"] == "comfy"
    assert saved["illustration_visual_states"] == {
        "Hero": {"visual_profile_id": "awakened"},
    }


@pytest.mark.asyncio
async def test_remote_hybrid_regeneration_updates_session_and_prompt(
    monkeypatch,
    tmp_path,
):
    session_id = "remote_hybrid_123456"
    prompt_id = "remote-hybrid-prompt"
    monkeypatch.setattr(pipeline, "SESSION_DIR", str(tmp_path / "sessions"))
    pipeline.create_session(session_id, "")
    pipeline.set_session_result(
        session_id,
        [{"slot": 0, "backup_name": "source-backup"}],
        [b"old-image"],
    )
    server.prompts[prompt_id] = {
        "status": "running",
        "save_node_id": "9",
        "image_bytes": None,
    }

    async def fake_regenerate(
        request,
        *,
        _body=None,
        _return_queue_result=False,
    ):
        assert request is None
        assert _body == {"name": "source-backup"}
        assert _return_queue_result is True
        return {
            "success": True,
            "image_bytes": b"new-image",
            "backup_name": "new-backup",
            "provider": "comfy",
            "fallback_used": True,
        }

    completed = []

    async def fake_complete(actual_prompt_id, save_node, filename):
        completed.append((actual_prompt_id, save_node, filename))
        server.prompts[actual_prompt_id]["status"] = "completed"

    monkeypatch.setattr(server, "handle_api_regenerate", fake_regenerate)
    monkeypatch.setattr(server, "complete_prompt_from_reschedule", fake_complete)

    try:
        result = await server.process_illustration_remote_regenerate(
            prompt_id,
            session_id,
            0,
            "source-backup",
        )

        assert result["success"] is True
        assert result["provider"] == "comfy"
        assert result["fallback_used"] is True
        assert pipeline.session_image_by_slot(session_id, 0) == b"new-image"
        assert (
            pipeline.session_item_by_slot(session_id, 0)["backup_name"]
            == "new-backup"
        )
        assert server.prompts[prompt_id]["image_bytes"] == b"new-image"
        assert completed and completed[0][0] == prompt_id
    finally:
        pipeline._SESSIONS.pop(session_id, None)
        server.prompts.pop(prompt_id, None)
