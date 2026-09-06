import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modes import llm_service
from queue_manager import QueueManager

LLM_SLOT_NUMBERS = tuple(range(1, 11))


def _slot_key(base: str, number: int) -> str:
    return base if number == 1 else f"{base}{number}"


def _config(**overrides):
    values = llm_service.get_config()
    for number in LLM_SLOT_NUMBERS:
        suffix = "" if number == 1 else str(number)
        values.update(
            {
                f"llm_service{suffix}": "openai",
                f"llm_model{suffix}": f"model-{number}",
                f"llm_api_key{suffix}": f"key-{number}",
                f"llm_url{suffix}": f"https://llm{number}.example",
                f"llm_stream{suffix}": False,
                f"llm_max_concurrency{suffix}": number,
                f"llm_stream_idle_timeout_seconds{suffix}": number + 10,
            }
        )
    values.update(overrides)
    return llm_service._ContextConfig(values)


@pytest.fixture(autouse=True)
def _clear_request_gates():
    llm_service._request_gates_by_loop.clear()
    yield
    llm_service._request_gates_by_loop.clear()


def test_routing_primary_max_concurrency_reads_selected_slot(monkeypatch):
    config = _config(
        llm_routing={
            "visual_profile_guide": {
                "primary": "llm3",
                "fallback": False,
            }
        }
    )
    monkeypatch.setattr(llm_service, "_current_config", config)

    assert llm_service.routing_primary_max_concurrency(
        "visual_profile_guide"
    ) == 3


@pytest.mark.asyncio
async def test_dispatch_enforces_each_slot_limit_independently(monkeypatch):
    monkeypatch.setattr(llm_service, "_current_config", _config())
    active = {f"llm{number}": 0 for number in LLM_SLOT_NUMBERS}
    maximum = {f"llm{number}": 0 for number in LLM_SLOT_NUMBERS}

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
        *(
            invoke(f"llm{number}")
            for number in LLM_SLOT_NUMBERS
            for _ in range(number + 1)
        ),
    )

    assert all(
        results.count(f"llm{number}") == number + 1
        for number in LLM_SLOT_NUMBERS
    )
    assert maximum == {
        f"llm{number}": number for number in LLM_SLOT_NUMBERS
    }


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


@pytest.mark.parametrize("number", LLM_SLOT_NUMBERS)
def test_stream_timeout_is_resolved_per_slot(monkeypatch, number):
    monkeypatch.setattr(llm_service, "_current_config", _config())
    slot = f"llm{number}"

    assert llm_service._stream_idle_timeout_seconds(slot) == number + 10

    token = llm_service._llm_slot_ctx.set(slot)
    try:
        assert llm_service._stream_http_timeout().read == number + 10
    finally:
        llm_service._llm_slot_ctx.reset(token)


def test_queue_worker_capacity_sums_only_configured_slots():
    manager = QueueManager()
    config = {}
    configured = {1, 2, 4, 5, 6, 7, 8, 9, 10}
    for number in LLM_SLOT_NUMBERS:
        suffix = "" if number == 1 else str(number)
        config[f"llm_model{suffix}"] = (
            f"model-{number}" if number in configured else ""
        )
        config[f"llm_max_concurrency{suffix}"] = number
    manager.get_config = lambda: config

    assert manager._target_llm_workers() == sum(configured)


def test_frontend_has_independent_controls_for_all_slots():
    html = (
        Path(__file__).resolve().parents[1] / "frontend" / "index.html"
    ).read_text(encoding="utf-8")

    # 슬롯 수는 백엔드 단일 소스(llm_service.LLM_SLOT_COUNT)와 같아야 한다.
    for n in LLM_SLOT_NUMBERS:
        suffix = "" if n == 1 else str(n)
        assert html.count(f'id="setting-llm-max-concurrency{suffix}"') == 1
        assert html.count(f'id="setting-llm-stream-idle-timeout{suffix}"') == 1


def test_llm_slot_count_and_ids_match_backend():
    # 슬롯 단일 소스가 프론트/백엔드/큐 매니저에서 일관되게 10개인지 확인.
    assert llm_service.LLM_SLOT_COUNT == 10
    assert llm_service.LLM_SLOT_IDS == tuple(
        f"llm{number}" for number in LLM_SLOT_NUMBERS
    )
    html = (
        Path(__file__).resolve().parents[1] / "frontend" / "index.html"
    ).read_text(encoding="utf-8")
    assert "constLLM_SLOTS=[1,2,3,4,5,6,7,8,9,10];" in html.replace(" ", "")
