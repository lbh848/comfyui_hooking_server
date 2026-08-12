"""_invoke_routed_with_retry 의 per-attempt 실패 콜백(on_attempt_failure) 검증.

목적: 캐릭터 메이커가 재시도 중 각 실패 시도를 자세히(lighbd_history.jsonl)에 개별
기록할 수 있도록, 라우트 계층이 (1) 매 실패 시도마다 콜백을 호출하고, (2) 성공 시엔
호출하지 않으며, (3) async 콜백을 await 하고, (4) 콜백 예외가 라우팅 흐름을 망가뜨리지
않는지 확인한다.
"""
import pytest

from modes import llm_service
from modes.llm_service import (
    _invoke_routed_with_retry,
    create_llm_execution_context,
)


async def _always_fails(slot):
    return "[LLM 실패] boom"


@pytest.mark.asyncio
async def test_on_attempt_failure_called_per_failed_attempt():
    """max_retries=1 → 총 2회 시도, 둘 다 실패 → 콜백이 attempt 1/2, 2/2 로 각각 호출."""
    calls = []

    def record(info):
        calls.append(dict(info))

    _result, accepted, _reason, _exc = await _invoke_routed_with_retry(
        "test_task",
        "primary",
        "llm1",
        1,  # max_retries → total_attempts = 2
        0,  # retry_delay_sec
        _always_fails,
        None,  # result_validator
        on_attempt_failure=record,
    )

    assert accepted is False
    assert len(calls) == 2
    assert [c["attempt"] for c in calls] == [1, 2]
    assert all(c["total_attempts"] == 2 for c in calls)
    assert all(c["phase"] == "primary" for c in calls)
    assert all(c["slot"] == "llm1" for c in calls)
    assert all(c["task_key"] == "test_task" for c in calls)
    assert all("boom" in c["reason"] for c in calls)


@pytest.mark.asyncio
async def test_on_attempt_failure_not_called_on_success():
    async def succeeds(slot):
        return "OK"

    called = []

    await _invoke_routed_with_retry(
        "test_task", "primary", "llm1", 2, 0, succeeds, None,
        on_attempt_failure=lambda info: called.append(info),
    )

    assert called == []


@pytest.mark.asyncio
async def test_on_attempt_failure_async_callback_is_awaited():
    """콜백이 코루틴을 반환하면 라우트가 await 해야 한다."""
    seen = []

    async def async_record(info):
        seen.append(info["attempt"])

    await _invoke_routed_with_retry(
        "test_task", "primary", "llm1", 0, 0, _always_fails, None,
        on_attempt_failure=async_record,
    )

    assert seen == [1]


@pytest.mark.asyncio
async def test_on_attempt_failure_exception_is_swallowed():
    """콜백이 예외를 던져도 라우팅 흐름은 살아있어야 한다(로깅이 호출을 망가뜨리면 안 됨)."""
    def bad_record(info):
        raise ValueError("callback boom")

    _result, accepted, _reason, _exc = await _invoke_routed_with_retry(
        "test_task", "primary", "llm1", 0, 0, _always_fails, None,
        on_attempt_failure=bad_record,
    )

    assert accepted is False


@pytest.mark.asyncio
async def test_execution_events_share_one_id_across_primary_and_fallback(monkeypatch):
    """전역 재시도/폴백의 모든 시도가 하나의 논리 실행 ID로 묶인다."""
    monkeypatch.setattr(llm_service, "_current_config", {"llm_routing": {}})
    monkeypatch.setattr(llm_service, "_routing_for", lambda _task: ("llm1", "llm2"))
    monkeypatch.setattr(
        llm_service,
        "_routing_retry_policy",
        lambda _task: {
            "max_retries": 0,
            "retry_delay_sec": 0.0,
            "fallback_max_retries": 0,
            "fallback_retry_delay_sec": 0.0,
        },
    )

    async def routed_slot(slot, *_args, **_kwargs):
        return "invalid primary" if slot == "llm1" else "valid fallback"

    monkeypatch.setattr(llm_service, "_call_routed_text_slot", routed_slot)
    observed = []
    context = create_llm_execution_context(
        "integration_task",
        call_name="통합 테스트",
        execution_id="exec-fixed",
        parent_execution_id="parent-fixed",
    )

    result = await llm_service.callLLMTaskResult(
        "integration_task",
        [{"role": "user", "content": "hello"}],
        result_validator=lambda value: (
            value == "valid fallback",
            "expected valid fallback",
        ),
        execution_context=context,
        execution_observer=observed.append,
    )

    assert result.accepted is True
    assert result.text == "valid fallback"
    assert result.final_phase == "fallback"
    assert result.final_slot == "llm2"
    assert [event["type"] for event in observed] == [
        "attempt_start",
        "attempt_failure",
        "attempt_start",
        "attempt_success",
        "execution_complete",
    ]
    assert {event["execution_id"] for event in observed} == {"exec-fixed"}
    assert {event["parent_execution_id"] for event in observed} == {"parent-fixed"}
    failed = observed[1]
    assert failed["raw_response"] == "invalid primary"
    assert failed["reason"] == "expected valid fallback"


@pytest.mark.asyncio
async def test_execution_result_and_legacy_wrapper_keep_exception_contract(monkeypatch):
    """내부 결과는 실패 원인을 보존하고 기존 공개 함수는 같은 예외를 다시 던진다."""
    monkeypatch.setattr(llm_service, "_current_config", {"llm_routing": {}})
    monkeypatch.setattr(llm_service, "_routing_for", lambda _task: ("llm1", None))
    monkeypatch.setattr(
        llm_service,
        "_routing_retry_policy",
        lambda _task: {
            "max_retries": 0,
            "retry_delay_sec": 0.0,
            "fallback_max_retries": 0,
            "fallback_retry_delay_sec": 0.0,
        },
    )

    async def raises(*_args, **_kwargs):
        raise TimeoutError("provider timeout body")

    monkeypatch.setattr(llm_service, "_call_routed_text_slot", raises)
    internal = await llm_service.callLLMTaskResult(
        "exception_task",
        [{"role": "user", "content": "hello"}],
    )

    assert internal.accepted is False
    assert isinstance(internal.exception, TimeoutError)
    assert "provider timeout body" in internal.reason
    with pytest.raises(TimeoutError, match="provider timeout body"):
        await llm_service.callLLMTask(
            "exception_task",
            [{"role": "user", "content": "hello"}],
        )


@pytest.mark.asyncio
async def test_vision_uses_same_execution_result_shape(monkeypatch):
    monkeypatch.setattr(llm_service, "_current_config", {"llm_routing": {}})
    monkeypatch.setattr(llm_service, "_routing_for", lambda _task: ("llm1", None))
    monkeypatch.setattr(
        llm_service,
        "_routing_retry_policy",
        lambda _task: {
            "max_retries": 0,
            "retry_delay_sec": 0.0,
            "fallback_max_retries": 0,
            "fallback_retry_delay_sec": 0.0,
        },
    )

    async def vision(*_args, **_kwargs):
        return "vision ok"

    monkeypatch.setattr(llm_service, "callLLMVision", vision)
    observed = []
    result = await llm_service.callLLMVisionTaskResult(
        "vision_task",
        [{"role": "user", "content": "look"}],
        image_b64="AA==",
        execution_id="vision-exec",
        execution_observer=observed.append,
    )

    assert result.accepted is True
    assert result.text == "vision ok"
    assert result.context.execution_id == "vision-exec"
    assert [event["type"] for event in observed] == [
        "attempt_start",
        "attempt_success",
        "execution_complete",
    ]


@pytest.mark.asyncio
async def test_auto_visual_format_repairs_do_not_trigger_llm_retry(monkeypatch):
    from modes.video_mode import validate_auto_visual_direction

    monkeypatch.setattr(llm_service, "_current_config", {"llm_routing": {}})
    monkeypatch.setattr(llm_service, "_routing_for", lambda _task: ("llm1", "llm2"))
    monkeypatch.setattr(
        llm_service,
        "_routing_retry_policy",
        lambda _task: {
            "max_retries": 2,
            "retry_delay_sec": 0.0,
            "fallback_max_retries": 2,
            "fallback_retry_delay_sec": 0.0,
        },
    )
    calls = []

    async def primary(*_args, **_kwargs):
        calls.append("llm1")
        return "```json\n['Picture 1: A quiet room.', '',]\n```"

    async def fallback(*_args, **_kwargs):
        calls.append("llm2")
        return '["Picture 1: fallback", "fallback direction"]'

    monkeypatch.setattr(llm_service, "callLLMVision", primary)
    monkeypatch.setattr(llm_service, "callLLMVision2", fallback)

    result = await llm_service.callLLMVisionTaskResult(
        "video_prompt_i2v",
        [{"role": "user", "content": "look"}],
        image_b64="AA==",
        result_validator=validate_auto_visual_direction,
    )

    assert calls == ["llm1"]
    assert result.accepted is True


@pytest.mark.asyncio
async def test_auto_visual_missing_context_retries_same_model(monkeypatch):
    from modes.video_mode import validate_auto_visual_direction

    monkeypatch.setattr(llm_service, "_current_config", {"llm_routing": {}})
    monkeypatch.setattr(llm_service, "_routing_for", lambda _task: ("llm1", "llm2"))
    monkeypatch.setattr(
        llm_service,
        "_routing_retry_policy",
        lambda _task: {
            "max_retries": 1,
            "retry_delay_sec": 0.0,
            "fallback_max_retries": 1,
            "fallback_retry_delay_sec": 0.0,
        },
    )
    responses = iter(
        [
            '["", "A valid direction without context."]',
            '["Picture 1: A recovered context.", "A recovered direction."]',
        ]
    )
    calls = []

    async def primary(*_args, **_kwargs):
        calls.append("llm1")
        return next(responses)

    async def fallback(*_args, **_kwargs):
        calls.append("llm2")
        return '["Picture 1: fallback", "fallback direction"]'

    monkeypatch.setattr(llm_service, "callLLMVision", primary)
    monkeypatch.setattr(llm_service, "callLLMVision2", fallback)

    result = await llm_service.callLLMVisionTaskResult(
        "video_prompt_i2v",
        [{"role": "user", "content": "look"}],
        image_b64="AA==",
        result_validator=validate_auto_visual_direction,
    )

    assert calls == ["llm1", "llm1"]
    assert result.accepted is True
