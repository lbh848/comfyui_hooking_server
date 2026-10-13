"""Provider budgets span slots and request types without changing slot settings."""
import asyncio

import pytest

from modes import llm_service


def shared_config():
    config = llm_service._ContextConfig({
        "llm_service": "vertex", "llm_model": "native-model",
        "llm_api_key": "shared-test-key", "llm_url": "https://provider.example/v1",
        "llm_max_concurrency": 1, "llm_routing": {},
    })
    for number in (2, 3, 4, 5, 10):
        config.update({
            f"llm_service{number}": "ollama-cloud" if number == 2 else "openai",
            f"llm_model{number}": f"model-{number}",
            f"llm_api_key{number}": "shared-test-key",
            f"llm_url{number}": "https://provider.example/v1/chat/completions",
            f"llm_max_concurrency{number}": 3,
            f"llm_stream{number}": False,
            f"llm_custom_headers{number}": "",
        })
    return config


@pytest.fixture(autouse=True)
def isolate_runtime(monkeypatch):
    llm_service._request_gates_by_loop.clear()
    llm_service._active_streams.clear()
    monkeypatch.setattr(llm_service, "_llm_log", lambda _message: None)
    monkeypatch.setattr(llm_service, "_log_history", lambda **_kwargs: None)
    yield
    llm_service._request_gates_by_loop.clear()
    llm_service._active_streams.clear()


async def wait_until(predicate):
    for _ in range(300):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("Expected provider state was not reached")


def test_provider_identity_uses_connection_not_service_alias_or_model():
    config = shared_config()
    config["llm_url2"] = "https://PROVIDER.example/v1/"
    members = llm_service._llm_provider_slots("llm3", config=config)
    assert members == ["llm2", "llm3", "llm4", "llm5", "llm10"]
    assert llm_service._llm_provider_max_concurrency("llm3", config=config) == 3
    assert "shared-test-key" not in llm_service._llm_provider_identity("llm3", config=config)
    # Native Vertex ignores the stale OpenAI URL; unconfigured slots must not
    # inherit the shared account's budget or lower it to their default of one.
    assert "llm1" not in members and "llm9" not in members


@pytest.mark.asyncio
@pytest.mark.parametrize("request_kinds", [
    [(3, "text")] * 3 + [(2, "stream"), (4, "vision"), (5, "text")],
    [(10, "stream"), (5, "vision"), (2, "text"), (3, "text"), (4, "stream")],
])
async def test_shared_provider_limits_mixed_actual_requests(monkeypatch, request_kinds):
    config = shared_config()
    monkeypatch.setattr(llm_service, "_current_config", config)
    release = asyncio.Event()
    active = 0
    maximum = 0
    seen = []

    async def enter(messages):
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        seen.append(llm_service._llm_slot_ctx.get())
        assert active <= 3
        await release.wait()

    async def text_provider(messages, service, model):
        nonlocal active
        await enter(messages)
        active -= 1
        return "accepted"

    async def stream_provider(messages, service, model):
        nonlocal active
        await enter(messages)
        yield {"type": "done", "text": "accepted", "elapsed": 0.1}
        active -= 1

    monkeypatch.setattr(llm_service, "_dispatch_unlimited", text_provider)
    monkeypatch.setattr(llm_service, "_dispatch_stream_unlimited", stream_provider)
    messages = [{"role": "user", "content": "context"}]

    async def request(number, kind):
        if kind == "stream":
            return await llm_service.callLLMTrackedStream(messages, slot=f"llm{number}")
        if kind == "vision":
            return await getattr(llm_service, f"callLLMVision{number}")(
                messages, image_b64="aW1hZ2U=", image_mime="image/png",
            )
        return await getattr(llm_service, f"callLLM{number}")(messages)

    tasks = [asyncio.create_task(request(number, kind)) for number, kind in request_kinds]
    try:
        await wait_until(lambda: active == 3)
        for _ in range(20):
            await asyncio.sleep(0)
        assert len(seen) == 3
        release.set()
        assert await asyncio.wait_for(asyncio.gather(*tasks), 3) == ["accepted"] * len(tasks)
        assert maximum == 3 and len(seen) == len(tasks)
        assert all(llm_service._request_gate(f"llm{n}").active == 0 for n, _ in request_kinds)
        assert llm_service._request_gate("llm3").provider_active == {}
    finally:
        release.set()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("difference", ["api_key", "url"])
async def test_distinct_provider_accounts_keep_independent_capacity(monkeypatch, difference):
    config = shared_config()
    config[f"llm_{difference}10"] = "other-key" if difference == "api_key" else "https://other.example/v1"
    config["llm_max_concurrency10"] = 2
    monkeypatch.setattr(llm_service, "_current_config", config)
    shared = llm_service._request_gate("llm3")
    independent = llm_service._request_gate("llm10")
    reservations = [(shared, await shared.acquire()) for _ in range(3)]
    reservations += [(independent, await independent.acquire()) for _ in range(2)]
    try:
        assert shared.active == 3 and independent.active == 2
        assert not shared.has_capacity_now() and not independent.has_capacity_now()
    finally:
        for gate, identity in reservations:
            await gate.release(identity)


@pytest.mark.asyncio
async def test_waiter_cancellation_and_connection_change_release_original_budget(monkeypatch):
    config = shared_config()
    monkeypatch.setattr(llm_service, "_current_config", config)
    gate = llm_service._request_gate("llm3")
    reservations = [await gate.acquire() for _ in range(2)]
    waiter_gate = llm_service._request_gate("llm2")
    other_identity = await waiter_gate.acquire()
    waiter = asyncio.create_task(waiter_gate.acquire())
    await asyncio.sleep(0)
    assert not waiter.done() and waiter_gate.active == 1
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    # A settings change must not release an older request into its new account.
    config["llm_api_key3"] = "new-account"
    new_identity = await gate.acquire()
    await gate.release(new_identity)
    assert len(gate.provider_active) == 1
    for identity in reservations:
        await gate.release(identity)
    await waiter_gate.release(other_identity)
    assert gate.active == 0 and gate.provider_active == {}


@pytest.mark.asyncio
async def test_configured_shared_limit_and_manual_retry_use_same_budget(monkeypatch):
    config = shared_config()
    config["llm_max_concurrency2"] = 2
    monkeypatch.setattr(llm_service, "_current_config", config)
    gate = llm_service._request_gate("llm3")
    other = llm_service._request_gate("llm10")
    first = await gate.acquire()
    second = await other.acquire()
    try:
        assert llm_service._llm_max_concurrency("llm3") == 3
        assert llm_service._llm_provider_max_concurrency("llm3") == 2
        assert not gate.has_capacity_now()
        assert await gate.try_acquire() is None
        llm_service.update_config({"llm_max_concurrency2": 3})
        retry = await gate.try_acquire()
        assert retry is not None
        await gate.release(retry)
    finally:
        await gate.release(first)
        await other.release(second)


@pytest.mark.asyncio
async def test_routing_retry_and_fallback_share_provider_budget(monkeypatch):
    config = shared_config()
    config["llm_routing"] = {"provider-test": {
        "primary": "llm3", "max_retries": 1, "retry_delay_sec": 0,
        "fallback": True, "fallback_target": "llm4", "fallback_max_retries": 0,
    }}
    monkeypatch.setattr(llm_service, "_current_config", config)
    blocker = llm_service._request_gate("llm2")
    held = [await blocker.acquire() for _ in range(2)]
    seen = []

    async def provider(messages, service, model):
        slot = llm_service._llm_slot_ctx.get()
        seen.append(slot)
        assert sum(blocker.provider_active.values()) == 3
        return "[LLM 실패] unavailable" if slot == "llm3" else "fallback accepted"

    monkeypatch.setattr(llm_service, "_dispatch_unlimited", provider)
    try:
        result = await llm_service.callLLMTask("provider-test", [{"role": "user", "content": "context"}])
        assert result == "fallback accepted"
        assert seen == ["llm3", "llm3", "llm4"]
        assert sum(blocker.provider_active.values()) == 2
    finally:
        for identity in held:
            await blocker.release(identity)


@pytest.mark.asyncio
async def test_live_parallel_retry_cannot_bypass_other_slots_provider_usage(monkeypatch):
    config = shared_config()
    monkeypatch.setattr(llm_service, "_current_config", config)
    blocker = llm_service._request_gate("llm2")
    held = [await blocker.acquire() for _ in range(2)]
    release = asyncio.Event()
    calls = 0

    async def provider(messages, service, model):
        nonlocal calls
        calls += 1
        yield {"type": "delta", "text": "partial", "elapsed": 0.1}
        await release.wait()
        yield {"type": "done", "text": "complete", "elapsed": 0.2}

    monkeypatch.setattr(llm_service, "_dispatch_stream_unlimited", provider)
    task = asyncio.create_task(llm_service.callLLMTrackedStream(
        [{"role": "user", "content": "context"}], slot="llm3",
    ))
    try:
        await wait_until(lambda: any(stream.get("text") == "partial" for stream in llm_service.get_active_streams()))
        original = llm_service.get_active_streams()[0]
        assert not original["parallel_retry_available"]
        accepted, message = llm_service.request_stream_control(original["stream_id"], "parallel_retry")
        assert not accepted and "프로바이더 3/3" in message
        assert calls == 1
        release.set()
        assert await asyncio.wait_for(task, 3) == "complete"
    finally:
        release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        for identity in held:
            await blocker.release(identity)
