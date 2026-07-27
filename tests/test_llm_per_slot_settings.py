import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modes import llm_service
from queue_manager import QueueManager


def _config(**overrides):
    values = llm_service.get_config()
    values.update(
        {
            "llm_service": "openai",
            "llm_model": "model-1",
            "llm_service2": "openai",
            "llm_model2": "model-2",
            "llm_service3": "openai",
            "llm_model3": "model-3",
            "llm_api_key": "key-1",
            "llm_api_key2": "key-2",
            "llm_api_key3": "key-3",
            "llm_url": "https://llm1.example",
            "llm_url2": "https://llm2.example",
            "llm_url3": "https://llm3.example",
            "llm_stream": False,
            "llm_stream2": False,
            "llm_stream3": False,
            "llm_max_concurrency": 1,
            "llm_max_concurrency2": 2,
            "llm_max_concurrency3": 3,
            "llm_stream_idle_timeout_seconds": 11,
            "llm_stream_idle_timeout_seconds2": 22,
            "llm_stream_idle_timeout_seconds3": 33,
        }
    )
    values.update(overrides)
    return llm_service._ContextConfig(values)


@pytest.fixture(autouse=True)
def _clear_request_gates():
    llm_service._request_gates_by_loop.clear()
    yield
    llm_service._request_gates_by_loop.clear()


@pytest.mark.asyncio
async def test_dispatch_enforces_each_slot_limit_independently(monkeypatch):
    monkeypatch.setattr(llm_service, "_current_config", _config())
    active = {"llm1": 0, "llm2": 0, "llm3": 0}
    maximum = {"llm1": 0, "llm2": 0, "llm3": 0}

    async def fake_openai(messages, model):
        slot = llm_service._llm_slot_ctx.get()
        active[slot] += 1
        maximum[slot] = max(maximum[slot], active[slot])
        await asyncio.sleep(0.02)
        active[slot] -= 1
        return slot

    monkeypatch.setattr(llm_service, "_call_openai_direct", fake_openai)

    async def invoke(slot):
        token = llm_service._llm_slot_ctx.set(slot)
        try:
            return await llm_service._dispatch([], "openai", f"{slot}-model")
        finally:
            llm_service._llm_slot_ctx.reset(token)

    results = await asyncio.gather(
        *(invoke("llm1") for _ in range(4)),
        *(invoke("llm2") for _ in range(4)),
        *(invoke("llm3") for _ in range(4)),
    )

    assert results.count("llm1") == 4
    assert results.count("llm2") == 4
    assert results.count("llm3") == 4
    assert maximum == {"llm1": 1, "llm2": 2, "llm3": 3}


@pytest.mark.asyncio
async def test_limit_increase_wakes_existing_waiter(monkeypatch):
    config = _config(llm_max_concurrency=1)
    monkeypatch.setattr(llm_service, "_current_config", config)
    first_started = asyncio.Event()
    second_started = asyncio.Event()
    release = asyncio.Event()
    call_count = 0

    async def fake_openai(messages, model):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            first_started.set()
        else:
            second_started.set()
        await release.wait()
        return "ok"

    monkeypatch.setattr(llm_service, "_call_openai_direct", fake_openai)
    first = asyncio.create_task(llm_service._dispatch([], "openai", "model"))
    await asyncio.wait_for(first_started.wait(), timeout=1)
    second = asyncio.create_task(llm_service._dispatch([], "openai", "model"))
    await asyncio.sleep(0)
    assert not second_started.is_set()

    llm_service.update_config({"llm_max_concurrency": 2})
    await asyncio.wait_for(second_started.wait(), timeout=1)
    release.set()
    await asyncio.gather(first, second)


@pytest.mark.asyncio
async def test_llm2_request_overlay_does_not_pollute_llm1(monkeypatch):
    monkeypatch.setattr(llm_service, "_current_config", _config())
    both_entered = asyncio.Event()
    entered = 0

    async def fake_dispatch(messages, service, model):
        nonlocal entered
        entered += 1
        if entered == 2:
            both_entered.set()
        await asyncio.wait_for(both_entered.wait(), timeout=1)
        return (
            llm_service._current_config.get("llm_api_key"),
            llm_service._current_config.get("llm_url"),
        )

    monkeypatch.setattr(llm_service, "_dispatch", fake_dispatch)
    llm1_result, llm2_result = await asyncio.gather(
        llm_service.callLLM([]),
        llm_service.callLLM2([]),
    )

    assert llm1_result == ("key-1", "https://llm1.example")
    assert llm2_result == ("key-2", "https://llm2.example")
    assert llm_service._current_config.get("llm_api_key") == "key-1"
    assert llm_service._current_config.get("llm_url") == "https://llm1.example"


def test_stream_timeout_is_resolved_per_slot(monkeypatch):
    monkeypatch.setattr(llm_service, "_current_config", _config())

    assert llm_service._stream_idle_timeout_seconds("llm1") == 11
    assert llm_service._stream_idle_timeout_seconds("llm2") == 22
    assert llm_service._stream_idle_timeout_seconds("llm3") == 33

    token = llm_service._llm_slot_ctx.set("llm2")
    try:
        assert llm_service._stream_http_timeout().read == 22
    finally:
        llm_service._llm_slot_ctx.reset(token)


def test_queue_worker_capacity_sums_only_configured_slots():
    manager = QueueManager()
    manager.get_config = lambda: {
        "llm_model": "model-1",
        "llm_model2": "model-2",
        "llm_model3": "",
        "llm_max_concurrency": 2,
        "llm_max_concurrency2": 4,
        "llm_max_concurrency3": 9,
    }

    assert manager._target_llm_workers() == 6


def test_frontend_has_independent_controls_for_all_slots():
    html = (
        Path(__file__).resolve().parents[1] / "frontend" / "index.html"
    ).read_text(encoding="utf-8")

    for suffix in ("", "2", "3"):
        assert html.count(f'id="setting-llm-max-concurrency{suffix}"') == 1
        assert html.count(f'id="setting-llm-stream-idle-timeout{suffix}"') == 1
