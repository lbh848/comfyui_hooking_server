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

    assert "illustration_call2_select" in routing
    assert "illustration_call2_plan" in routing
    assert "illustration_call2" in routing
    assert "illustration_call2_keyvis" in routing
    assert "illustration_call2_authority_audit" in routing
    assert "json_mode" not in routing["illustration_call2_authority_audit"]
    assert "illustration_scene_curate" not in routing


def test_call2_split_routes_are_registered_in_frontend_settings() -> None:
    frontend = (
        Path(__file__).resolve().parents[1] / "frontend" / "index.html"
    ).read_text(encoding="utf-8")

    assert "{ key: 'illustration_call2_select'" in frontend
    assert "{ key: 'illustration_call2_plan'" in frontend
    assert "{ key: 'illustration_call2'," in frontend
    assert "{ key: 'illustration_call2_keyvis'" in frontend
    assert "{ key: 'illustration_call2_authority_audit'" in frontend
    assert "삽화 CALL2-PLAN" in frontend
    assert "삽화 CALL2-DETAIL" in frontend
    assert "삽화 CALL2-KEYVIS" in frontend
    assert "삽화 CALL2-AUTHORITY-AUDIT" in frontend


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
        "illustration_profile_context_translate",
        "illustration_profile_resolve",
        "illustration_original_asset",
        "illustration_original_asset_recovery",
        "illustration_call1_backtranslate",
        "illustration_call1",
        "illustration_call2_select",
        "illustration_call2_plan",
        "illustration_call2_keyvis",
        "illustration_call2",
        "illustration_call2_authority_audit",
        "illustration_call2_fix",
        "illustration_call3",
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
    assert merged["illustration_call2_authority_audit"]["primary"] == legacy_route["primary"]
    assert merged["illustration_call2_authority_audit"]["fallback"] == legacy_route["fallback"]
    assert "json_mode" not in merged["illustration_call2_authority_audit"]
    assert merged["illustration_call2_plan"] is not merged["illustration_call2"]
    assert merged["illustration_call2_keyvis"] is not merged["illustration_call2"]
    assert merged["illustration_call2_authority_audit"] is not merged["illustration_call2"]


def test_explicit_call2_split_route_overrides_legacy_inheritance() -> None:
    merged = server._merge_llm_routing_config({
        "llm_routing": {
            "illustration_call2": {"primary": "llm2", "max_retries": 4},
            "illustration_call2_plan": {"primary": "llm3", "max_retries": 1},
            "illustration_call2_authority_audit": {
                "primary": "llm5",
                "max_retries": 0,
                "json_mode": True,
            },
        },
    })

    assert merged["illustration_call2"]["primary"] == "llm2"
    assert merged["illustration_call2_plan"]["primary"] == "llm3"
    assert merged["illustration_call2_plan"]["max_retries"] == 1
    assert merged["illustration_call2_keyvis"]["primary"] == "llm2"
    assert merged["illustration_call2_keyvis"]["max_retries"] == 4
    assert merged["illustration_call2_authority_audit"]["primary"] == "llm5"
    assert merged["illustration_call2_authority_audit"]["max_retries"] == 0


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
    assert normalized["illustration_call2_authority_audit"]["primary"] == "llm2"
    assert normalized["illustration_call2_authority_audit"]["fallback"] is True
    assert "json_mode" not in normalized["illustration_call2_authority_audit"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("call_name", "expected_task_key", "expected_group_id"),
    [
        ("CALL2-SELECT", "illustration_call2_select", "call2_select"),
        ("CALL2-PLAN", "illustration_call2_plan", "call2_plan"),
        ("CALL2-PLAN 2/3", "illustration_call2_plan", "call2_plan"),
        (
            "CALL2-DETAIL 1/1 [FULL c1/6]",
            "illustration_call2",
            "call2_detail",
        ),
        (
            "CALL2-DETAIL 2/3 [PARTIAL c2/6]",
            "illustration_call2",
            "call2_detail",
        ),
        ("CALL2-KEYVIS", "illustration_call2_keyvis", "call2_keyvis"),
        (
            "CALL2-AUTHORITY-AUDIT",
            "illustration_call2_authority_audit",
            "call2_authority_audit",
        ),
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


@pytest.mark.parametrize("normalizer", [server._merge_llm_routing_config,
                                      server._normalize_llm_routing_for_save])
def test_selection_initial_route_preserves_plan_and_explicit_selection_is_independent(normalizer):
    routes = {"illustration_call2": {"primary": "llm2", "max_retries": 2},
              "illustration_call2_plan": {"primary": "llm4", "max_retries": 3}}
    def normalize(value):
        return normalizer({"llm_routing": value} if normalizer is server._merge_llm_routing_config else value)
    inherited = normalize(routes)
    assert inherited["illustration_call2_select"]["primary"] == "llm4"
    assert inherited["illustration_call2_select"]["max_retries"] == 3
    explicit = normalize({**routes, "illustration_call2_select": {"primary": "llm7", "max_retries": 1}})
    assert explicit["illustration_call2_select"]["primary"] == "llm7"
    assert explicit["illustration_call2_select"]["max_retries"] == 1
    assert explicit["illustration_call2_plan"]["primary"] == "llm4"


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["CALL2-SELECT", "CALL2-PLAN 1/2"])
async def test_selection_and_parallel_plan_terminal_failure_is_recorded(monkeypatch, stage):
    records = []
    async def fail_call(_task_key, _messages, **_kwargs):
        raise RuntimeError("configured route failed")
    monkeypatch.setattr(pipeline.llm_service, "callLLMTask", fail_call)
    monkeypatch.setattr(pipeline.lighbd_service, "_log_lighbd_history", records.append)
    with pytest.raises(RuntimeError, match="configured route failed"):
        await pipeline._call_pipeline_llm(stage, [{"role": "user", "content": "source"}])
    assert records[-1]["task_key"] == ("illustration_call2_select" if stage == "CALL2-SELECT"
                                       else "illustration_call2_plan")
    assert records[-1]["call_name"] == stage
    assert records[-1]["status"] == "error"
    assert records[-1]["execution_id"] and records[-1]["input"]


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["CALL2-SELECT", "CALL2-PLAN 2/3"])
async def test_selection_and_parallel_plan_routing_retry_failure_preserves_raw_history(monkeypatch, stage):
    records = []
    async def retry_call(_task_key, _messages, **kwargs):
        observer = kwargs["execution_observer"]
        await observer({"type": "attempt_failure", "phase": "primary", "slot": "llm4",
                        "attempt": 1, "total_attempts": 2, "raw_response": "invalid original response",
                        "reason": "assigned scenes missing"})
        await observer({"type": "attempt_success", "phase": "primary", "slot": "llm4",
                        "attempt": 2, "total_attempts": 2, "raw_response": "valid retried result"})
        return "valid retried result"
    monkeypatch.setattr(pipeline.llm_service, "callLLMTask", retry_call)
    monkeypatch.setattr(pipeline.lighbd_service, "_log_lighbd_history", records.append)
    assert await pipeline._call_pipeline_llm(stage, [{"role": "user", "content": "source"}]) == "valid retried result"
    assert [record["status"] for record in records] == ["error", "ok"]
    assert records[0]["output"] == "invalid original response"
    assert records[0]["error"] == "assigned scenes missing"
    assert records[0]["llm_slot"] == "llm4"
    assert records[0]["attempt"] == 1 and records[1]["attempt"] == 2
    assert all(record["task_key"] == pipeline._CALL_TASK_KEYS[stage.split()[0]] for record in records)
    assert all(record["call_name"] == stage and record["execution_id"] for record in records)
