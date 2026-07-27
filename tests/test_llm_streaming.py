import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modes import llm_service


def _test_config():
    config = llm_service.get_config()
    config.update({
        "llm_service": "openai",
        "llm_model": "model-1",
        "llm_service2": "openai",
        "llm_model2": "model-2",
        "llm_service3": "openai",
        "llm_model3": "model-3",
        "llm_stream": False,
        "llm_stream2": False,
        "llm_stream3": False,
        "llm_stream_idle_timeout_seconds": 90,
        "llm_routing": {},
    })
    return config


async def _fake_stream(messages, service, model):
    assert messages == [{"role": "user", "content": "hello"}]
    yield {"type": "start", "service": service, "model": model}
    yield {"type": "delta", "text": "안", "elapsed": 0.1, "ttft": 0.1}
    yield {"type": "delta", "text": "녕", "elapsed": 0.2, "ttft": 0.1}
    yield {
        "type": "done",
        "text": "안녕",
        "completion_tokens": 2,
        "elapsed": 0.2,
        "tps": 10.0,
        "ttft": 0.1,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("slot", "toggle_key", "call_name"),
    [
        ("llm1", "llm_stream", "callLLM"),
        ("llm2", "llm_stream2", "callLLM2"),
        ("llm3", "llm_stream3", "callLLM3"),
    ],
)
async def test_each_llm_toggle_uses_real_stream_and_forwards_deltas(
    monkeypatch, slot, toggle_key, call_name
):
    config = _test_config()
    config[toggle_key] = True
    monkeypatch.setattr(llm_service, "_current_config", config)
    monkeypatch.setattr(llm_service, "_dispatch_stream", _fake_stream)

    events = []

    async def notify(event):
        events.append(event)

    monkeypatch.setattr(llm_service, "_stream_notify_func", notify)

    result = await getattr(llm_service, call_name)(
        [{"role": "user", "content": "hello"}]
    )

    assert result == "안녕"
    assert [event["type"] for event in events] == ["start", "delta", "delta", "done"]
    assert all(event["llm_slot"] == slot for event in events)
    assert len({event["stream_id"] for event in events}) == 1


@pytest.mark.asyncio
async def test_disabled_toggle_keeps_non_stream_call(monkeypatch):
    config = _test_config()
    monkeypatch.setattr(llm_service, "_current_config", config)

    async def dispatch(messages, service, model):
        return "완료 응답"

    async def unexpected_stream(*args, **kwargs):
        raise AssertionError("스트리밍 토글이 꺼졌는데 스트림 경로가 호출됨")
        yield

    monkeypatch.setattr(llm_service, "_dispatch", dispatch)
    monkeypatch.setattr(llm_service, "_dispatch_stream", unexpected_stream)

    result = await llm_service.callLLM([{"role": "user", "content": "hello"}])

    assert result == "완료 응답"


@pytest.mark.asyncio
async def test_task_stream_event_contains_task_metadata(monkeypatch):
    config = _test_config()
    config["llm_stream2"] = True
    config["llm_routing"] = {
        "unit_task": {"primary": "llm2", "fallback_target": None}
    }
    monkeypatch.setattr(llm_service, "_current_config", config)
    monkeypatch.setattr(llm_service, "_dispatch_stream", _fake_stream)

    events = []

    async def notify(event):
        events.append(event)

    monkeypatch.setattr(llm_service, "_stream_notify_func", notify)

    result = await llm_service.callLLMTask(
        "unit_task", [{"role": "user", "content": "hello"}]
    )

    assert result == "안녕"
    assert events
    assert all(event["task_key"] == "unit_task" for event in events)
    assert all(event["call_name"] == "unit_task" for event in events)
    assert all(event["llm_slot"] == "llm2" for event in events)


@pytest.mark.asyncio
async def test_task_metadata_sink_captures_stream_usage(monkeypatch):
    """스트리밍 callLLMTask 가 done 이벤트의 usage 토큰을 metadata_sink 에 채운다."""
    config = _test_config()
    config["llm_stream"] = True
    config["llm_routing"] = {
        "unit_task": {"primary": "llm1", "fallback_target": None}
    }
    monkeypatch.setattr(llm_service, "_current_config", config)

    async def fake_stream(messages, service, model):
        yield {"type": "start", "service": service, "model": model}
        yield {"type": "delta", "text": "안녕"}
        yield {
            "type": "done",
            "text": "안녕",
            "completion_tokens": 42,
            "prompt_tokens": 128,
            "elapsed": 0.5,
            "tps": 84.0,
        }

    monkeypatch.setattr(llm_service, "_dispatch_stream", fake_stream)

    async def notify(_event):
        return None

    monkeypatch.setattr(llm_service, "_stream_notify_func", notify)

    sink: dict = {}
    result = await llm_service.callLLMTask(
        "unit_task",
        [{"role": "user", "content": "hello"}],
        metadata_sink=sink,
    )

    assert result == "안녕"
    assert sink.get("completion_tokens") == 42
    assert sink.get("prompt_tokens") == 128
    assert sink.get("tps") == 84.0


@pytest.mark.asyncio
async def test_task_metadata_sink_falls_back_to_approx_offline(monkeypatch):
    """비스트리밍 callLLMTask 는 usage 를 못 얻으므로 sink 를 근사치로 채운다."""
    config = _test_config()
    config["llm_stream"] = False
    config["llm_routing"] = {
        "unit_task": {"primary": "llm1", "fallback_target": None}
    }
    monkeypatch.setattr(llm_service, "_current_config", config)

    async def dispatch(messages, service, model):
        return "ABCDEFGHIJ"  # 10자 → _approx_tokens = max(1, 10//3) = 3

    monkeypatch.setattr(llm_service, "_dispatch", dispatch)

    sink: dict = {}
    result = await llm_service.callLLMTask(
        "unit_task",
        [{"role": "user", "content": "hello"}],
        metadata_sink=sink,
    )

    assert result == "ABCDEFGHIJ"
    assert sink.get("completion_tokens") == 3
    assert sink.get("prompt_tokens") >= 1


@pytest.mark.asyncio
async def test_task_stream_inherits_pipeline_display_call_name(monkeypatch):
    config = _test_config()
    config["llm_stream2"] = True
    config["llm_routing"] = {
        "illustration_call2": {"primary": "llm2", "fallback_target": None}
    }
    monkeypatch.setattr(llm_service, "_current_config", config)
    monkeypatch.setattr(llm_service, "_dispatch_stream", _fake_stream)
    events = []

    async def notify(event):
        events.append(event)

    monkeypatch.setattr(llm_service, "_stream_notify_func", notify)
    token = llm_service._stream_metadata_ctx.set({
        "task_key": "illustration_call2",
        "call_name": "CALL2-PLAN",
    })
    try:
        result = await llm_service.callLLMTask(
            "illustration_call2",
            [{"role": "user", "content": "hello"}],
        )
    finally:
        llm_service._stream_metadata_ctx.reset(token)

    assert result == "안녕"
    assert events
    assert all(event["task_key"] == "illustration_call2" for event in events)
    assert all(event["call_name"] == "CALL2-PLAN" for event in events)


@pytest.mark.asyncio
async def test_task_stream_observer_receives_request_local_partial_lengths(monkeypatch):
    config = _test_config()
    config["llm_stream"] = True
    config["llm_routing"] = {
        "unit_task": {"primary": "llm1", "fallback_target": None}
    }
    monkeypatch.setattr(llm_service, "_current_config", config)
    monkeypatch.setattr(llm_service, "_dispatch_stream", _fake_stream)

    async def notify(_event):
        return None

    monkeypatch.setattr(llm_service, "_stream_notify_func", notify)
    observed = []

    result = await llm_service.callLLMTask(
        "unit_task",
        [{"role": "user", "content": "hello"}],
        stream_observer=observed.append,
    )

    assert result == "안녕"
    assert observed[0]["type"] == "request_mode"
    assert observed[0]["streaming"] is True
    assert observed[1]["type"] == "stream_open"
    assert [
        event["partial_length"]
        for event in observed
        if event["type"] == "delta"
    ] == [1, 2]
    assert observed[-1]["type"] == "done"
    assert observed[-1]["partial_text"] == "안녕"


@pytest.mark.asyncio
async def test_task_stream_observer_marks_non_streaming_request_without_partial_events(monkeypatch):
    config = _test_config()
    config["llm_routing"] = {
        "unit_task": {"primary": "llm1", "fallback_target": None}
    }
    monkeypatch.setattr(llm_service, "_current_config", config)

    async def dispatch(_messages, _service, _model):
        return "완료"

    monkeypatch.setattr(llm_service, "_dispatch", dispatch)
    observed = []

    result = await llm_service.callLLMTask(
        "unit_task",
        [{"role": "user", "content": "hello"}],
        stream_observer=observed.append,
    )

    assert result == "완료"
    assert observed == [{
        "type": "request_mode",
        "task_key": "unit_task",
        "llm_slot": "llm1",
        "streaming": False,
    }]


@pytest.mark.asyncio
async def test_task_retries_none_empty_and_whitespace_responses(monkeypatch):
    config = _test_config()
    config["llm_routing"] = {
        "unit_task": {
            "primary": "llm1",
            "fallback": False,
            "max_retries": 3,
            "retry_delay_sec": 2.5,
            "fallback_max_retries": 0,
            "fallback_retry_delay_sec": 0,
        }
    }
    monkeypatch.setattr(llm_service, "_current_config", config)

    responses = iter([None, "", "   \n\t", "완료"])
    calls = []
    sleeps = []

    async def dispatch(messages, service, model):
        calls.append((service, model))
        return next(responses)

    async def fake_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(llm_service, "_dispatch", dispatch)
    monkeypatch.setattr(llm_service.asyncio, "sleep", fake_sleep)

    result = await llm_service.callLLMTask(
        "unit_task", [{"role": "user", "content": "hello"}]
    )

    assert result == "완료"
    assert len(calls) == 4
    assert sleeps == [2.5, 2.5, 2.5]


@pytest.mark.asyncio
async def test_task_returns_explicit_failure_after_blank_responses_are_exhausted(monkeypatch):
    config = _test_config()
    config["llm_routing"] = {
        "unit_task": {
            "primary": "llm1",
            "fallback": False,
            "max_retries": 1,
            "retry_delay_sec": 0,
            "fallback_max_retries": 0,
            "fallback_retry_delay_sec": 0,
        }
    }
    monkeypatch.setattr(llm_service, "_current_config", config)

    async def dispatch(messages, service, model):
        return "  \n"

    monkeypatch.setattr(llm_service, "_dispatch", dispatch)

    result = await llm_service.callLLMTask(
        "unit_task", [{"role": "user", "content": "hello"}]
    )

    assert result.startswith("[LLM 실패]")
    assert "응답이 비어 있음" in result


@pytest.mark.asyncio
async def test_task_exhausts_primary_then_uses_independent_fallback_policy(monkeypatch):
    config = _test_config()
    config["llm_routing"] = {
        "unit_task": {
            "primary": "llm1",
            "fallback": True,
            "fallback_target": "llm2",
            "max_retries": 1,
            "retry_delay_sec": 0.25,
            "fallback_max_retries": 2,
            "fallback_retry_delay_sec": 0.5,
        }
    }
    monkeypatch.setattr(llm_service, "_current_config", config)

    slots = []
    sleeps = []
    fallback_results = iter(["[LLM 실패] 일시 오류", " ", "폴백 완료"])

    async def fake_slot(slot, messages, model=None, json_mode=False):
        slots.append(slot)
        if slot == "llm1":
            return ""
        return next(fallback_results)

    async def fake_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(llm_service, "_call_routed_text_slot", fake_slot)
    monkeypatch.setattr(llm_service.asyncio, "sleep", fake_sleep)

    result = await llm_service.callLLMTask(
        "unit_task", [{"role": "user", "content": "hello"}]
    )

    assert result == "폴백 완료"
    assert slots == ["llm1", "llm1", "llm2", "llm2", "llm2"]
    assert sleeps == [0.25, 0.5, 0.5]


@pytest.mark.asyncio
async def test_task_retries_response_validator_failures(monkeypatch):
    config = _test_config()
    config["llm_routing"] = {
        "unit_task": {
            "primary": "llm1",
            "fallback": False,
            "max_retries": 1,
            "retry_delay_sec": 0,
            "fallback_max_retries": 0,
            "fallback_retry_delay_sec": 0,
        }
    }
    monkeypatch.setattr(llm_service, "_current_config", config)

    responses = iter(["not-json", '{"ok": true}'])
    calls = []

    async def dispatch(messages, service, model):
        calls.append(model)
        return next(responses)

    monkeypatch.setattr(llm_service, "_dispatch", dispatch)

    result = await llm_service.callLLMTask(
        "unit_task",
        [{"role": "user", "content": "hello"}],
        result_validator=lambda value: (value.startswith("{"), "JSON 아님"),
    )

    assert result == '{"ok": true}'
    assert calls == ["model-1", "model-1"]


@pytest.mark.asyncio
async def test_task_retries_raised_call_exception(monkeypatch):
    config = _test_config()
    config["llm_routing"] = {
        "unit_task": {
            "primary": "llm1",
            "fallback": False,
            "max_retries": 1,
            "retry_delay_sec": 0,
            "fallback_max_retries": 0,
            "fallback_retry_delay_sec": 0,
        }
    }
    monkeypatch.setattr(llm_service, "_current_config", config)

    calls = 0

    async def dispatch(messages, service, model):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("temporary failure")
        return "복구됨"

    monkeypatch.setattr(llm_service, "_dispatch", dispatch)

    result = await llm_service.callLLMTask(
        "unit_task", [{"role": "user", "content": "hello"}]
    )

    assert result == "복구됨"
    assert calls == 2


@pytest.mark.asyncio
async def test_vision_task_uses_same_primary_and_fallback_retry_policy(monkeypatch):
    config = _test_config()
    config["llm_routing"] = {
        "vision_task": {
            "primary": "llm1",
            "fallback": True,
            "fallback_target": "llm2",
            "max_retries": 0,
            "retry_delay_sec": 0,
            "fallback_max_retries": 1,
            "fallback_retry_delay_sec": 0,
        }
    }
    monkeypatch.setattr(llm_service, "_current_config", config)

    slots = []
    fallback_responses = iter(["", "비전 완료"])

    async def fake_vision1(*args, **kwargs):
        slots.append("llm1")
        return " "

    async def fake_vision2(*args, **kwargs):
        slots.append("llm2")
        return next(fallback_responses)

    monkeypatch.setattr(llm_service, "callLLMVision", fake_vision1)
    monkeypatch.setattr(llm_service, "callLLMVision2", fake_vision2)

    result = await llm_service.callLLMVisionTask(
        "vision_task",
        [{"role": "user", "content": "inspect"}],
        image_b64="aW1hZ2U=",
    )

    assert result == "비전 완료"
    assert slots == ["llm1", "llm2", "llm2"]


@pytest.mark.asyncio
async def test_task_parallel_routes_isolate_llm2_and_llm3_config(monkeypatch):
    config = _test_config()
    config.update({
        "llm_api_key": "key-1",
        "llm_api_key2": "key-2",
        "llm_api_key3": "key-3",
        "llm_url": "https://llm1.invalid",
        "llm_url2": "https://llm2.invalid",
        "llm_url3": "https://llm3.invalid",
        "llm_routing": {
            "task2": {"primary": "llm2", "fallback": False},
            "task3": {"primary": "llm3", "fallback": False},
        },
    })
    monkeypatch.setattr(llm_service, "_current_config", llm_service._ContextConfig(config))

    both_started = asyncio.Event()
    started = 0

    async def dispatch(messages, service, model):
        nonlocal started
        before = (
            llm_service._current_config.get("llm_api_key"),
            llm_service._current_config.get("llm_url"),
        )
        started += 1
        if started == 2:
            both_started.set()
        await both_started.wait()
        after = (
            llm_service._current_config.get("llm_api_key"),
            llm_service._current_config.get("llm_url"),
        )
        return f"{model}|{before}|{after}"

    monkeypatch.setattr(llm_service, "_dispatch", dispatch)
    result2, result3 = await asyncio.gather(
        llm_service.callLLMTask("task2", [{"role": "user", "content": "two"}]),
        llm_service.callLLMTask("task3", [{"role": "user", "content": "three"}]),
    )

    assert "model-2|('key-2', 'https://llm2.invalid')|('key-2', 'https://llm2.invalid')" == result2
    assert "model-3|('key-3', 'https://llm3.invalid')|('key-3', 'https://llm3.invalid')" == result3
    assert llm_service._current_config.get("llm_api_key") == "key-1"
    assert llm_service._current_config.get("llm_url") == "https://llm1.invalid"


@pytest.mark.asyncio
async def test_llm3_stream_api_yields_provider_deltas_without_buffering(monkeypatch):
    config = _test_config()
    monkeypatch.setattr(llm_service, "_current_config", config)
    monkeypatch.setattr(llm_service, "_dispatch_stream", _fake_stream)

    events = [
        event
        async for event in llm_service.callLLM3Stream(
            [{"role": "user", "content": "hello"}], log_history=False
        )
    ]

    assert [event["type"] for event in events] == ["start", "delta", "delta", "done"]
    assert "".join(event.get("text", "") for event in events if event["type"] == "delta") == "안녕"


async def _wait_for_active_stream(predicate=lambda _stream: True):
    for _ in range(200):
        streams = llm_service.get_active_streams()
        if streams and predicate(streams[0]):
            return streams[0]
        await asyncio.sleep(0)
    raise AssertionError("활성 스트림이 제한 시간 안에 등록되지 않음")


@pytest.mark.asyncio
async def test_active_stream_snapshot_tracks_partial_text_and_cleans_up(monkeypatch):
    config = _test_config()
    config["llm_stream"] = True
    monkeypatch.setattr(llm_service, "_current_config", config)
    llm_service._active_streams.clear()
    release = asyncio.Event()

    async def controlled_stream(messages, service, model):
        yield {"type": "start", "service": service, "model": model}
        yield {"type": "delta", "text": "부분", "elapsed": 0.1, "ttft": 0.1}
        await release.wait()
        yield {
            "type": "done",
            "text": "부분 완료",
            "completion_tokens": 2,
            "elapsed": 0.2,
            "tps": 10.0,
            "ttft": 0.1,
        }

    monkeypatch.setattr(llm_service, "_dispatch_stream", controlled_stream)
    task = asyncio.create_task(
        llm_service.callLLM([{"role": "user", "content": "hello"}])
    )
    snapshot = await _wait_for_active_stream(lambda stream: stream["text"] == "부분")

    assert snapshot["active"] is True
    assert snapshot["llm_slot"] == "llm1"
    assert snapshot["text"] == "부분"
    assert not any(key.startswith("_") for key in snapshot)

    release.set()
    assert await task == "부분 완료"
    assert llm_service.get_active_streams() == []


@pytest.mark.asyncio
async def test_manual_retry_closes_old_stream_and_starts_new_stream(monkeypatch):
    config = _test_config()
    config["llm_stream"] = True
    monkeypatch.setattr(llm_service, "_current_config", config)
    llm_service._active_streams.clear()
    attempts = 0
    events = []

    async def retryable_stream(messages, service, model):
        nonlocal attempts
        attempts += 1
        yield {"type": "start", "service": service, "model": model}
        if attempts == 1:
            yield {"type": "delta", "text": "이전 부분", "elapsed": 0.1, "ttft": 0.1}
            await asyncio.Future()
        yield {"type": "delta", "text": "재시도 완료", "elapsed": 0.1, "ttft": 0.1}
        yield {
            "type": "done",
            "text": "재시도 완료",
            "completion_tokens": 2,
            "elapsed": 0.2,
            "tps": 10.0,
            "ttft": 0.1,
        }

    async def notify(event):
        events.append(event)

    monkeypatch.setattr(llm_service, "_dispatch_stream", retryable_stream)
    monkeypatch.setattr(llm_service, "_stream_notify_func", notify)
    task = asyncio.create_task(
        llm_service.callLLM([{"role": "user", "content": "hello"}])
    )
    first = await _wait_for_active_stream(lambda stream: stream["text"] == "이전 부분")

    assert llm_service.request_stream_control(first["stream_id"], "retry") == (True, "retry")
    assert await task == "재시도 완료"
    assert attempts == 2
    assert llm_service.get_active_streams() == []
    cancelled = [event for event in events if event["type"] == "cancelled"]
    assert len(cancelled) == 1
    assert cancelled[0]["reason"] == "retry"
    assert len({event["stream_id"] for event in events}) == 2


@pytest.mark.asyncio
async def test_use_partial_requires_and_passes_task_validator(monkeypatch):
    config = _test_config()
    config["llm_stream"] = True
    config["llm_routing"] = {
        "unit_task": {
            "primary": "llm1",
            "fallback": False,
            "max_retries": 0,
            "retry_delay_sec": 0,
        }
    }
    monkeypatch.setattr(llm_service, "_current_config", config)
    llm_service._active_streams.clear()

    async def partial_stream(messages, service, model):
        yield {"type": "start", "service": service, "model": model}
        yield {"type": "delta", "text": "검증 가능한 부분", "elapsed": 0.1, "ttft": 0.1}
        await asyncio.Future()

    monkeypatch.setattr(llm_service, "_dispatch_stream", partial_stream)
    task = asyncio.create_task(
        llm_service.callLLMTask(
            "unit_task",
            [{"role": "user", "content": "hello"}],
            result_validator=lambda value: (value.startswith("검증 가능한"), "형식 오류"),
        )
    )
    stream = await _wait_for_active_stream(
        lambda item: item["text"] == "검증 가능한 부분"
    )

    assert llm_service.request_stream_control(stream["stream_id"], "use_partial") == (
        True,
        "use_partial",
    )
    result = await task
    assert result == "검증 가능한 부분"
    assert isinstance(result, llm_service.PartialStreamText)
    assert llm_service.get_active_streams() == []


@pytest.mark.asyncio
async def test_idle_timeout_ends_stream_and_preserves_partial_in_error_event(monkeypatch):
    config = _test_config()
    config["llm_stream"] = True
    monkeypatch.setattr(llm_service, "_current_config", config)
    monkeypatch.setattr(llm_service, "_stream_idle_timeout_seconds", lambda: 0.01)
    llm_service._active_streams.clear()
    events = []

    async def stalled_stream(messages, service, model):
        yield {"type": "start", "service": service, "model": model}
        yield {"type": "delta", "text": "남은 부분", "elapsed": 0.1, "ttft": 0.1}
        await asyncio.Future()

    async def notify(event):
        events.append(event)

    monkeypatch.setattr(llm_service, "_dispatch_stream", stalled_stream)
    monkeypatch.setattr(llm_service, "_stream_notify_func", notify)

    result = await llm_service.callLLM([{"role": "user", "content": "hello"}])

    assert result.startswith("[LLM 실패]")
    timeout_event = next(event for event in events if event.get("termination_reason") == "idle_timeout")
    assert timeout_event["partial_text"] == "남은 부분"
    assert llm_service.get_active_streams() == []


@pytest.mark.asyncio
async def test_openai_compat_finish_reason_ends_without_done_sentinel(monkeypatch):
    config = _test_config()
    monkeypatch.setattr(llm_service, "_current_config", config)
    consumed_lines = []

    class FakeResponse:
        status_code = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def aiter_lines(self):
            lines = [
                'data: {"choices":[{"delta":{"content":"완료"},"finish_reason":null}]}',
                'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
            ]
            for line in lines:
                consumed_lines.append(line)
                yield line
            raise AssertionError("finish_reason 뒤의 연결 종료를 기다리면 안 됨")

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def stream(self, *args, **kwargs):
            return FakeResponse()

    monkeypatch.setattr(llm_service.httpx, "AsyncClient", lambda **kwargs: FakeClient())
    events = [
        event
        async for event in llm_service._stream_openai_compat(
            [{"role": "user", "content": "hello"}],
            "model-1",
            "https://example.invalid/v1/chat/completions",
        )
    ]

    assert [event["type"] for event in events] == ["start", "delta", "done"]
    assert events[-1]["text"] == "완료"
    assert len(consumed_lines) == 2
