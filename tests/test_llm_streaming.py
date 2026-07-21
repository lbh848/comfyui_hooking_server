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
