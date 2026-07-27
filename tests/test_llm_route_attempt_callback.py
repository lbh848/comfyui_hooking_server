"""_invoke_routed_with_retry 의 per-attempt 실패 콜백(on_attempt_failure) 검증.

목적: 캐릭터 메이커가 재시도 중 각 실패 시도를 자세히(lighbd_history.jsonl)에 개별
기록할 수 있도록, 라우트 계층이 (1) 매 실패 시도마다 콜백을 호출하고, (2) 성공 시엔
호출하지 않으며, (3) async 콜백을 await 하고, (4) 콜백 예외가 라우팅 흐름을 망가뜨리지
않는지 확인한다.
"""
import pytest

from modes.llm_service import _invoke_routed_with_retry


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
