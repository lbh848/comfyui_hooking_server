import asyncio
import json
from pathlib import Path
import time

import pytest

import illustration_flow
import server
from modes import illustration_context_pipeline as pipeline
from modes import visual_profile_translation_cache as translation_cache
from modes.visual_profiles import (
    build_natural_profile_catalog,
    cards_to_character_profiles,
)


def _profiles(selection_one="첫 번째 모습이 유지되는 동안 선택한다."):
    return {
        "Riko": cards_to_character_profiles("Riko", [
            {
                "id": "ordinary",
                "label": "카드 1",
                "selection_guide": selection_one,
                "visual_context": "갈색 머리와 푸른 눈의 평상시 모습.",
                "appearance": ["brown hair"],
                "default_outfit": ["school uniform"],
            },
            {
                "id": "awakened",
                "label": "카드 2",
                "selection_guide": "변신이 완료되어 각성 형태가 유지되는 동안 선택한다.",
                "visual_context": "푸른 장발과 금빛 눈, 흰 망토를 착용한 모습.",
                "appearance": ["blue hair"],
                "default_outfit": ["white cape"],
            },
        ])
    }


def _patch_cache_paths(monkeypatch, tmp_path):
    cache_path = tmp_path / "visual_profile_translation_cache.json"
    backup_dir = tmp_path / "backups"
    monkeypatch.setattr(translation_cache, "CACHE_FILE", str(cache_path))
    monkeypatch.setattr(translation_cache, "BACKUP_DIR", str(backup_dir))
    return cache_path, backup_dir


def _translation_request(messages):
    return json.loads(messages[-1]["content"].split("\n\n", 1)[1])


@pytest.mark.asyncio
async def test_profile_translation_cache_miss_hit_and_single_field_invalidation(
    monkeypatch,
    tmp_path,
):
    cache_path, backup_dir = _patch_cache_paths(monkeypatch, tmp_path)
    calls = []

    async def fake_call(call_name, messages, _stream_notify, **kwargs):
        assert call_name == pipeline.PROFILE_CONTEXT_TRANSLATE_CALL_NAME
        assert kwargs["json_mode"] is True
        requested = _translation_request(messages)
        calls.append(requested)
        return json.dumps({
            "translations": [
                {
                    "ref": item["ref"],
                    "english": f"English translation: {item['source']}",
                }
                for item in requested
            ]
        }, ensure_ascii=False)

    monkeypatch.setattr(pipeline, "_call_pipeline_llm", fake_call)

    first, first_summary = await pipeline.prepare_profile_context_translations(
        _profiles(),
        cache_namespace="demo-bot",
    )

    assert len(calls) == 1
    assert len(calls[0]) == 4
    assert first_summary == {
        "cache_namespace": "demo-bot",
        "profile_count": 2,
        "cache_hits": 0,
        "cache_misses": 4,
        "empty_fields": 0,
        "translated_fields": 4,
        "fallback_fields": 0,
        "cache_saved": True,
    }
    assert first["Riko"]["profiles"][0]["selection_guide_english"].startswith(
        "English translation:"
    )
    assert cache_path.is_file()

    second, second_summary = await pipeline.prepare_profile_context_translations(
        _profiles(),
        cache_namespace="demo-bot",
    )

    assert len(calls) == 1
    assert second_summary["cache_hits"] == 4
    assert second_summary["cache_misses"] == 0
    assert second_summary["cache_saved"] is False
    assert second["Riko"]["profiles"][1]["visual_context_english"].startswith(
        "English translation:"
    )

    changed_source = "첫 번째 모습으로 되돌아온 뒤 그 상태가 유지되는 동안 선택한다."
    changed, changed_summary = await pipeline.prepare_profile_context_translations(
        _profiles(selection_one=changed_source),
        cache_namespace="demo-bot",
    )

    assert len(calls) == 2
    assert [item["source"] for item in calls[1]] == [changed_source]
    assert changed_summary["cache_hits"] == 3
    assert changed_summary["cache_misses"] == 1
    assert changed_summary["translated_fields"] == 1
    assert changed["Riko"]["profiles"][0]["selection_guide_english"].endswith(
        changed_source
    )
    assert list(backup_dir.glob("*.json"))


@pytest.mark.asyncio
async def test_same_text_on_distinct_profiles_keeps_distinct_cache_entries(
    monkeypatch,
    tmp_path,
):
    cache_path, _backup_dir = _patch_cache_paths(monkeypatch, tmp_path)
    profiles = _profiles(selection_one="같은 문장")
    profiles["Riko"]["profiles"][1]["selection_guide"] = "같은 문장"
    for profile in profiles["Riko"]["profiles"]:
        profile["visual_context"] = ""

    async def fake_call(_call_name, messages, _stream_notify, **_kwargs):
        requested = _translation_request(messages)
        return json.dumps({
            "translations": [
                {"ref": item["ref"], "english": "Same sentence."}
                for item in requested
            ]
        })

    monkeypatch.setattr(pipeline, "_call_pipeline_llm", fake_call)

    _translated, summary = await pipeline.prepare_profile_context_translations(
        profiles,
        cache_namespace="demo-bot",
    )
    stored = json.loads(cache_path.read_text(encoding="utf-8"))

    assert summary["cache_misses"] == 2
    assert len(stored["entries"]) == 2
    assert {
        entry["profile_id"] for entry in stored["entries"].values()
    } == {"ordinary", "awakened"}


@pytest.mark.asyncio
async def test_concurrent_requests_share_one_cache_miss_translation(
    monkeypatch,
    tmp_path,
):
    _patch_cache_paths(monkeypatch, tmp_path)
    call_count = 0

    async def fake_call(_call_name, messages, _stream_notify, **_kwargs):
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0)
        requested = _translation_request(messages)
        return json.dumps({
            "translations": [
                {"ref": item["ref"], "english": f"EN: {item['source']}"}
                for item in requested
            ]
        }, ensure_ascii=False)

    monkeypatch.setattr(pipeline, "_call_pipeline_llm", fake_call)

    results = await asyncio.gather(
        pipeline.prepare_profile_context_translations(
            _profiles(),
            cache_namespace="demo-bot",
        ),
        pipeline.prepare_profile_context_translations(
            _profiles(),
            cache_namespace="demo-bot",
        ),
    )
    summaries = [summary for _profiles_result, summary in results]

    assert call_count == 1
    assert sorted(summary["cache_misses"] for summary in summaries) == [0, 4]
    assert sorted(summary["cache_hits"] for summary in summaries) == [0, 4]


@pytest.mark.asyncio
async def test_cache_hit_request_does_not_wait_for_another_translation(
    monkeypatch,
    tmp_path,
):
    _patch_cache_paths(monkeypatch, tmp_path)

    async def immediate_call(_call_name, messages, _stream_notify, **_kwargs):
        requested = _translation_request(messages)
        return json.dumps({
            "translations": [
                {"ref": item["ref"], "english": f"EN: {item['source']}"}
                for item in requested
            ]
        }, ensure_ascii=False)

    monkeypatch.setattr(pipeline, "_call_pipeline_llm", immediate_call)
    await pipeline.prepare_profile_context_translations(
        _profiles(),
        cache_namespace="cached-bot",
    )

    translation_started = asyncio.Event()
    release_translation = asyncio.Event()

    async def delayed_call(_call_name, messages, _stream_notify, **_kwargs):
        translation_started.set()
        await release_translation.wait()
        requested = _translation_request(messages)
        return json.dumps({
            "translations": [
                {"ref": item["ref"], "english": f"EN: {item['source']}"}
                for item in requested
            ]
        }, ensure_ascii=False)

    monkeypatch.setattr(pipeline, "_call_pipeline_llm", delayed_call)
    translating = asyncio.create_task(
        pipeline.prepare_profile_context_translations(
            _profiles(),
            cache_namespace="uncached-bot",
        )
    )
    try:
        await asyncio.wait_for(translation_started.wait(), timeout=1)
        translated, summary = await asyncio.wait_for(
            pipeline.prepare_profile_context_translations(
                _profiles(),
                cache_namespace="cached-bot",
            ),
            timeout=1,
        )
    finally:
        release_translation.set()
        await translating

    assert summary["cache_hits"] == 4
    assert summary["cache_misses"] == 0
    assert translated["Riko"]["profiles"][0]["selection_guide_english"].startswith(
        "EN:"
    )


@pytest.mark.asyncio
async def test_translation_failure_uses_original_for_this_run_without_caching(
    monkeypatch,
    tmp_path,
):
    cache_path, _backup_dir = _patch_cache_paths(monkeypatch, tmp_path)

    async def fail_call(_call_name, _messages, _stream_notify, **_kwargs):
        raise RuntimeError("synthetic translation outage")

    monkeypatch.setattr(pipeline, "_call_pipeline_llm", fail_call)

    translated, summary = await pipeline.prepare_profile_context_translations(
        _profiles(),
        cache_namespace="demo-bot",
    )

    assert summary["fallback_fields"] == 4
    assert summary["cache_saved"] is False
    assert "selection_guide_english" not in translated["Riko"]["profiles"][0]
    assert not cache_path.exists()


def test_natural_profile_catalog_prefers_cached_english_fields():
    profiles = _profiles()
    ordinary, awakened = profiles["Riko"]["profiles"]
    ordinary["selection_guide_english"] = "Use while Riko remains in her ordinary form."
    ordinary["visual_context_english"] = "Brown hair and blue eyes."
    awakened["selection_guide_english"] = "Use after the transformation is complete."
    awakened["visual_context_english"] = "Long blue hair, golden eyes, and a white cape."

    catalog = build_natural_profile_catalog(profiles)

    assert "Use while Riko remains in her ordinary form." in catalog
    assert "Long blue hair, golden eyes, and a white cape." in catalog
    assert "첫 번째 모습이 유지되는 동안 선택한다." not in catalog
    assert "푸른 장발과 금빛 눈" not in catalog


@pytest.mark.asyncio
async def test_profile_resolution_lazily_translates_after_character_resolution(
    monkeypatch,
    tmp_path,
):
    _patch_cache_paths(monkeypatch, tmp_path)
    calls = []
    english_by_source = {
        "첫 번째 모습이 유지되는 동안 선택한다.": "Use while the ordinary form persists.",
        "갈색 머리와 푸른 눈의 평상시 모습.": "Brown hair and blue eyes.",
        "변신이 완료되어 각성 형태가 유지되는 동안 선택한다.": (
            "Use after the transformation is complete and the awakened form persists."
        ),
        "푸른 장발과 금빛 눈, 흰 망토를 착용한 모습.": (
            "Long blue hair, golden eyes, and a white cape."
        ),
    }

    async def fake_pipeline_call(call_name, messages, _stream_notify, **_kwargs):
        prompt = "\n".join(message["content"] for message in messages)
        calls.append(call_name)
        if call_name == "CHARACTER-RESOLVE":
            return json.dumps({
                "characters": [{"name": "Riko"}, {"name": "Bob"}],
                "uncertainties": [],
            })
        if call_name == pipeline.PROFILE_CONTEXT_TRANSLATE_CALL_NAME:
            requested = _translation_request(messages)
            assert {item["source"] for item in requested} == set(english_by_source)
            return json.dumps({
                "translations": [
                    {
                        "ref": item["ref"],
                        "english": english_by_source[item["source"]],
                    }
                    for item in requested
                ]
            })
        assert call_name == "PROFILE-RESOLVE"
        assert "Use while the ordinary form persists." in prompt
        assert "Long blue hair, golden eyes, and a white cape." in prompt
        assert "첫 번째 모습이 유지되는 동안 선택한다." not in prompt
        assert "푸른 장발과 금빛 눈" not in prompt
        return json.dumps({
            "characters": [{
                "name": "Riko",
                "profile_timeline": [{
                    "at": "START",
                    "profile_ref": "[2]",
                    "reason": "The transformation is complete, so [2] is active.",
                }],
            }],
            "uncertainties": [],
        })

    monkeypatch.setattr(pipeline, "_call_pipeline_llm", fake_pipeline_call)
    profiles = _profiles()
    profiles["Bob"] = cards_to_character_profiles("Bob", [{
        "id": "ordinary",
        "label": "카드 1",
        "selection_guide": "밥의 기본 모습.",
        "visual_context": "검은 머리.",
        "appearance": ["black hair"],
        "default_outfit": ["suit"],
    }])

    _output, result = await pipeline.resolve_profiles_before_generation(
        payload={"chats": [{"role": "char", "data": "리코가 변신을 마쳤다."}]},
        toggles={"profile_resolve_enabled": True},
        history_plan=None,
        visual_profiles=profiles,
        profile_translation_namespace="demo-bot",
    )

    assert calls == [
        "CHARACTER-RESOLVE",
        pipeline.PROFILE_CONTEXT_TRANSLATE_CALL_NAME,
        "PROFILE-RESOLVE",
    ]
    assert result["initial_visual_bases"][0]["target_visual_profile_id"] == "awakened"


def test_profile_context_translation_is_registered_for_routing_and_workflow():
    route = server.DEFAULT_CONFIG["llm_routing"][
        pipeline.PROFILE_CONTEXT_TRANSLATE_TASK_KEY
    ]
    frontend = (
        Path(__file__).parents[1] / "frontend" / "index.html"
    ).read_text(encoding="utf-8")
    flow_frontend = (
        Path(__file__).parents[1] / "frontend" / "illustration_flow.js"
    ).read_text(encoding="utf-8")

    assert route["primary"] == "llm1"
    assert route["json_mode"] is True
    assert (
        pipeline._CALL_TASK_KEYS[pipeline.PROFILE_CONTEXT_TRANSLATE_CALL_NAME]
        == pipeline.PROFILE_CONTEXT_TRANSLATE_TASK_KEY
    )
    assert pipeline._CALL_QUEUE_SUBTASK_GROUPS[
        pipeline.PROFILE_CONTEXT_TRANSLATE_CALL_NAME
    ][0] == "profile_context_translation"
    assert "key: 'illustration_profile_context_translate'" in frontend
    assert "JSON · 캐시 미스 시" in frontend
    assert "label === 'PROFILE-CONTEXT-CACHE'" in flow_frontend
    assert "label.startsWith('PROFILE-CONTEXT-TRANSLATE')" in flow_frontend
    assert "PROFILE-CONTEXT-CACHE" in Path(
        pipeline.__file__
    ).read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_cache_hit_summary_is_visible_as_a_workflow_stage():
    run = {
        "id": "profile-cache-flow",
        "kind": "illustration",
        "label": "translation cache test",
        "status": "processing",
        "cancel_requested": False,
        "created_at": time.time(),
        "updated_at": time.time(),
        "revision": 0,
        "nodes": {},
        "_active_llm_tasks": set(),
    }
    summary = {
        "cache_namespace": "demo-bot",
        "profile_count": 2,
        "cache_hits": 4,
        "cache_misses": 0,
        "empty_fields": 0,
        "translated_fields": 0,
        "fallback_fields": 0,
        "cache_saved": False,
    }
    token_run = illustration_flow._run.set(run)
    token_frontier = illustration_flow._frontier.set(("character-resolve",))
    try:
        recorded = await pipeline._record_profile_context_cache_summary(summary)
    finally:
        illustration_flow._frontier.reset(token_frontier)
        illustration_flow._run.reset(token_run)

    node = next(iter(run["nodes"].values()))
    assert recorded == summary
    assert node["label"] == "PROFILE-CONTEXT-CACHE"
    assert node["status"] == "completed"
    assert node["layout_group"] == "profile_context_translation"
    assert node["dependencies"] == ["character-resolve"]
    assert node["output"]["cache_hits"] == 4
    assert node["output"]["cache_misses"] == 0


@pytest.mark.asyncio
async def test_translation_call_records_task_key_in_lb_details_and_flow_events(
    monkeypatch,
):
    records = []
    events = []
    messages = [{"role": "user", "content": "translate profile context"}]

    async def fake_llm_call(task_key, actual_messages, **kwargs):
        assert task_key == pipeline.PROFILE_CONTEXT_TRANSLATE_TASK_KEY
        assert actual_messages == messages
        kwargs["metadata_sink"].update({
            "completion_tokens": 7,
            "prompt_tokens": 11,
            "elapsed": 0.25,
            "tps": 28.0,
        })
        await kwargs["execution_observer"]({
            "type": "attempt_success",
            "phase": "primary",
            "slot": "llm3",
            "attempt": 1,
            "total_attempts": 1,
            "attempt_id": "translation-attempt-1",
        })
        return '{"translations":[{"ref":"text_1","english":"English."}]}'

    async def notify(event):
        events.append(event)

    monkeypatch.setattr(pipeline.llm_service, "callLLMTask", fake_llm_call)
    monkeypatch.setattr(pipeline.lighbd_service, "_log_lighbd_history", records.append)

    result = await pipeline._call_pipeline_llm(
        pipeline.PROFILE_CONTEXT_TRANSLATE_CALL_NAME,
        messages,
        notify,
        json_mode=True,
    )

    assert result.startswith('{"translations"')
    assert [event["type"] for event in events] == ["start", "done"]
    assert {
        event["queue_subtask"]["group_id"] for event in events
    } == {"profile_context_translation"}
    assert len(records) == 1
    assert records[0]["task_key"] == pipeline.PROFILE_CONTEXT_TRANSLATE_TASK_KEY
    assert records[0]["call_name"] == pipeline.PROFILE_CONTEXT_TRANSLATE_CALL_NAME
    assert records[0]["status"] == "ok"
    assert records[0]["llm_slot"] == "llm3"
    assert records[0]["input"] == messages
    assert records[0]["output"] == result
    assert records[0]["completion_tokens"] == 7
    assert records[0]["prompt_tokens"] == 11
