import copy
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import server
from modes import illustration_context_pipeline as pipeline


def test_call2_split_routes_are_registered_in_backend_defaults() -> None:
    routing = server.DEFAULT_CONFIG["llm_routing"]

    assert "illustration_call2_plan" in routing
    assert "illustration_call2" in routing
    assert "illustration_call2_keyvis" in routing


def test_call2_split_routes_are_registered_in_frontend_settings() -> None:
    frontend = (
        Path(__file__).resolve().parents[1] / "frontend" / "index.html"
    ).read_text(encoding="utf-8")

    assert "{ key: 'illustration_call2_plan'" in frontend
    assert "{ key: 'illustration_call2'," in frontend
    assert "{ key: 'illustration_call2_keyvis'" in frontend
    assert "삽화 CALL2-PLAN" in frontend
    assert "삽화 CALL2-DETAIL" in frontend
    assert "삽화 CALL2-KEYVIS" in frontend


def test_illustration_routes_follow_runtime_call_order() -> None:
    frontend = (
        Path(__file__).resolve().parents[1] / "frontend" / "index.html"
    ).read_text(encoding="utf-8")
    task_block = frontend.split("const LLM_ROUTING_TASKS = [", 1)[1].split(
        "const LLM_ROUTING_MODALITIES = [",
        1,
    )[0]
    illustration_keys = [
        key
        for key, group in re.findall(
            r"\{ key: '([^']+)'[^\n]+group: '([^']+)'",
            task_block,
        )
        if group == "illustration_pipeline"
    ]

    assert illustration_keys == [
        "illustration_character_resolve",
        "illustration_profile_resolve",
        "illustration_original_asset",
        "illustration_original_asset_recovery",
        "illustration_call1_backtranslate",
        "illustration_call1",
        "illustration_call2_plan",
        "illustration_call2_keyvis",
        "illustration_call2",
        "illustration_call2_fix",
        "illustration_call3",
        "illustration_call3_subtitle",
        "illustration_multi_char_mask",
    ]


def test_call2_split_routes_inherit_legacy_detail_route_when_missing() -> None:
    legacy_route = {
        "primary": "llm2",
        "fallback": True,
        "fallback_target": "llm4",
        "max_retries": 3,
        "fallback_max_retries": 2,
        "retry_delay_sec": 1.5,
        "fallback_retry_delay_sec": 4.0,
    }

    merged = server._merge_llm_routing_config({
        "llm_routing": {"illustration_call2": copy.deepcopy(legacy_route)},
    })

    assert merged["illustration_call2"] == legacy_route
    assert merged["illustration_call2_plan"] == legacy_route
    assert merged["illustration_call2_keyvis"] == legacy_route
    assert merged["illustration_call2_plan"] is not merged["illustration_call2"]
    assert merged["illustration_call2_keyvis"] is not merged["illustration_call2"]


def test_explicit_call2_split_route_overrides_legacy_inheritance() -> None:
    merged = server._merge_llm_routing_config({
        "llm_routing": {
            "illustration_call2": {"primary": "llm2", "max_retries": 4},
            "illustration_call2_plan": {"primary": "llm3", "max_retries": 1},
        },
    })

    assert merged["illustration_call2"]["primary"] == "llm2"
    assert merged["illustration_call2_plan"]["primary"] == "llm3"
    assert merged["illustration_call2_plan"]["max_retries"] == 1
    assert merged["illustration_call2_keyvis"]["primary"] == "llm2"
    assert merged["illustration_call2_keyvis"]["max_retries"] == 4


def test_legacy_only_save_payload_preserves_call2_split_inheritance() -> None:
    normalized = server._normalize_llm_routing_for_save({
        "illustration_call2": {
            "primary": "llm2",
            "fallback": True,
            "fallback_target": "llm4",
            "max_retries": 3,
            "fallback_max_retries": 2,
            "retry_delay_sec": 1.5,
            "fallback_retry_delay_sec": 4.0,
        },
    })

    assert normalized["illustration_call2_plan"] == normalized["illustration_call2"]
    assert normalized["illustration_call2_keyvis"] == normalized["illustration_call2"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("call_name", "expected_task_key", "expected_group_id"),
    [
        ("CALL2-PLAN", "illustration_call2_plan", "call2_plan"),
        ("CALL2-DETAIL 1/2", "illustration_call2", "call2_detail"),
        ("CALL2-KEYVIS", "illustration_call2_keyvis", "call2_keyvis"),
    ],
)
async def test_call2_split_routes_reach_queue_and_history(
    monkeypatch,
    call_name,
    expected_task_key,
    expected_group_id,
) -> None:
    records = []
    events = []

    async def fake_call(task_key, _messages, **_kwargs):
        assert task_key == expected_task_key
        metadata = pipeline.llm_service._stream_metadata_ctx.get({})
        assert metadata["task_key"] == expected_task_key
        assert metadata["call_name"] == call_name
        return "ok"

    async def fake_notify(event):
        events.append(event)

    monkeypatch.setattr(pipeline.llm_service, "callLLMTask", fake_call)
    monkeypatch.setattr(pipeline.lighbd_service, "_log_lighbd_history", records.append)

    result = await pipeline._call_pipeline_llm(
        call_name,
        [{"role": "user", "content": "route me"}],
        fake_notify,
    )

    assert result == "ok"
    assert [event["type"] for event in events] == ["start", "done"]
    assert {event["queue_subtask"]["group_id"] for event in events} == {
        expected_group_id
    }
    assert len(records) == 1
    assert records[0]["task_key"] == expected_task_key
    assert records[0]["call_name"] == call_name
    assert records[0]["status"] == "ok"
