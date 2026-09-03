"""LLM 시스템의 실제 HTTP 경계를 포함한 격리형 E2E 감사 테스트.

외부 API나 운영 config/history를 건드리지 않는다. 테스트 안에서만 로컬
OpenAI 호환 제공자를 띄우고 LLM1~5 슬롯, 라우팅 재시도/폴백, 테스트 SSE
엔드포인트를 검증한다.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import re
import sys
import time
from contextlib import asynccontextmanager
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import pytest
from aiohttp import ClientSession, web

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import queue_manager as queue_manager_module
import server
from modes import (
    embedding_service,
    illustration_context_pipeline,
    lighbd_service,
    llm_service,
)

_REAL_NOTIFY_FRONTEND = server.notify_frontend


def _parse_sse(raw: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    normalized = raw.replace("\r\n", "\n")
    for frame in normalized.split("\n\n"):
        if not frame.strip():
            continue
        event_type = "message"
        data_parts: list[str] = []
        for line in frame.splitlines():
            if line.startswith("event:"):
                event_type = line[6:].strip()
            elif line.startswith("data:"):
                data_parts.append(line[5:].strip())
        if data_parts:
            events.append((event_type, json.loads("".join(data_parts))))
    return events


@asynccontextmanager
async def _running_app(app: web.Application):
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    sockets = getattr(site._server, "sockets", None) or []
    if not sockets:
        raise RuntimeError("테스트 HTTP 서버의 listen socket을 찾지 못했습니다")
    port = sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        await runner.cleanup()


def _isolated_config(**overrides) -> llm_service._ContextConfig:
    values = llm_service.get_config()
    values.update(
        {
            "llm_service": "openai",
            "llm_model": "slot-1",
            "llm_api_key": "test-key-1",
            "llm_url": "",
            "llm_stream": False,
            "llm_max_concurrency": 2,
            "llm_stream_idle_timeout_seconds": 90,
            "llm_routing": {},
        }
    )
    for slot in range(2, llm_service.LLM_SLOT_COUNT + 1):
        values.update(
            {
                f"llm_service{slot}": "openai",
                f"llm_model{slot}": f"slot-{slot}",
                f"llm_api_key{slot}": f"test-key-{slot}",
                f"llm_url{slot}": "",
                f"llm_stream{slot}": False,
                f"llm_max_concurrency{slot}": 2,
                f"llm_stream_idle_timeout_seconds{slot}": 90,
            }
        )
    values.update(overrides)
    return llm_service._ContextConfig(values)


class _ImmediateQueue:
    """HTTP E2E에서 런타임 핸들러를 즉시 실행하는 결정적 큐 대역."""

    def __init__(self):
        self.added: list[dict] = []

    async def add_item(
        self,
        item_type,
        label,
        params,
        priority=10,
        runtime_handler=None,
        **_kwargs,
    ):
        item = SimpleNamespace(
            id=f"test-{len(self.added) + 1}",
            type=item_type,
            label=label,
            params=params,
        )
        item.completion_future = asyncio.get_running_loop().create_future()
        self.added.append(
            {
                "item_type": item_type,
                "label": label,
                "params": params,
                "priority": priority,
                "runtime_handler": runtime_handler,
            }
        )
        try:
            result = await runtime_handler(item)
            item.completion_future.set_result(result)
        except Exception as exc:
            item.completion_future.set_exception(exc)
        return item


@pytest.fixture(autouse=True)
def _clear_llm_runtime_state(monkeypatch):
    llm_service._request_gates_by_loop.clear()
    llm_service._active_streams.clear()
    monkeypatch.setattr(llm_service, "_llm_log", lambda _message: None)
    monkeypatch.setattr(llm_service, "_log_history", lambda **_kwargs: None)
    monkeypatch.setattr(
        lighbd_service,
        "_log_lighbd_history",
        lambda _record: None,
    )
    monkeypatch.setattr(server, "queue_manager", _ImmediateQueue())

    async def quiet_notify(_event_type, _data=None):
        return None

    monkeypatch.setattr(server, "notify_frontend", quiet_notify)
    yield
    llm_service._request_gates_by_loop.clear()
    llm_service._active_streams.clear()


@pytest.mark.asyncio
async def test_llm1_to_llm5_reach_their_own_openai_compatible_slot(monkeypatch):
    requests: list[dict] = []

    async def completion(request: web.Request) -> web.Response:
        body = await request.json()
        requests.append(
            {
                "model": body.get("model"),
                "authorization": request.headers.get("Authorization"),
            }
        )
        model = str(body.get("model") or "")
        return web.json_response(
            {"choices": [{"message": {"content": f"ok:{model}"}}]}
        )

    provider = web.Application()
    provider.router.add_post("/v1/chat/completions", completion)
    async with _running_app(provider) as base_url:
        cfg = _isolated_config(llm_url=base_url)
        for slot in range(2, llm_service.LLM_SLOT_COUNT + 1):
            cfg[f"llm_url{slot}"] = base_url
        monkeypatch.setattr(llm_service, "_current_config", cfg)

        messages = [{"role": "user", "content": "slot smoke"}]
        calls = [
            llm_service.callLLM,
            llm_service.callLLM2,
            llm_service.callLLM3,
            llm_service.callLLM4,
            llm_service.callLLM5,
        ]
        results = [await call(messages) for call in calls]

    assert results == [f"ok:slot-{slot}" for slot in range(1, 6)]
    assert [item["model"] for item in requests] == [
        f"slot-{slot}" for slot in range(1, 6)
    ]
    assert [item["authorization"] for item in requests] == [
        f"Bearer test-key-{slot}" for slot in range(1, 6)
    ]


@pytest.mark.asyncio
async def test_vertex_openai_base64_crosses_http_and_sse_boundaries(monkeypatch):
    plain_response = "vertex-openai-base64-e2e-ok"
    encoded_response = base64.b64encode(plain_response.encode("utf-8")).decode("ascii")
    requests: list[dict] = []

    async def completion(request: web.Request) -> web.StreamResponse:
        body = await request.json()
        messages = body.get("messages") or []
        requests.append(
            {
                "authorization": request.headers.get("Authorization"),
                "messages": messages,
                "response_format": body.get("response_format"),
                "stream": bool(body.get("stream")),
            }
        )
        assert "Base64-Encoded Instruction Protocol" in messages[0]["content"]
        assert base64.b64decode(messages[1]["content"]).decode("utf-8") == "실제 HTTP 경계"

        if not body.get("stream"):
            return web.json_response(
                {"choices": [{"message": {"content": encoded_response}}]}
            )

        response = web.StreamResponse(
            status=200,
            headers={"Content-Type": "text/event-stream"},
        )
        await response.prepare(request)
        for start in range(0, len(encoded_response), 5):
            chunk = encoded_response[start:start + 5]
            payload = {"choices": [{"delta": {"content": chunk}}]}
            await response.write(f"data: {json.dumps(payload)}\n\n".encode("utf-8"))
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response

    provider = web.Application()
    provider.router.add_post("/v1/chat/completions", completion)
    async with _running_app(provider) as base_url:
        cfg = _isolated_config(
            llm_service="vertex-openai",
            llm_model="gemini-e2e",
            llm_url=base_url,
            llm_gemini_base64=True,
            llm_stream=False,
        )
        monkeypatch.setattr(llm_service, "_current_config", cfg)
        monkeypatch.setattr(llm_service, "_get_vertex_key_path", lambda: "test-vertex.json")

        async def fake_access_token(_key_path):
            return "vertex-e2e-token"

        monkeypatch.setattr(llm_service, "_get_vertex_access_token", fake_access_token)
        messages = [{"role": "user", "content": "실제 HTTP 경계"}]
        sync_result = await llm_service.callLLM(messages)
        cfg["llm_stream"] = True
        stream_result = await llm_service.callLLM(messages)

    assert sync_result == plain_response
    assert stream_result == plain_response
    assert [item["stream"] for item in requests] == [False, True]
    assert all(item["authorization"] == "Bearer vertex-e2e-token" for item in requests)
    assert all(item["response_format"] is None for item in requests)


@pytest.mark.asyncio
async def test_route_retries_http_failure_then_uses_independent_fallback(monkeypatch):
    request_models: list[str] = []

    async def completion(request: web.Request) -> web.Response:
        body = await request.json()
        model = str(body.get("model") or "")
        request_models.append(model)
        if model == "primary-503":
            return web.json_response(
                {"error": {"message": "temporary overload"}},
                status=503,
            )
        return web.json_response(
            {"choices": [{"message": {"content": "fallback-ok"}}]}
        )

    provider = web.Application()
    provider.router.add_post("/v1/chat/completions", completion)
    async with _running_app(provider) as base_url:
        cfg = _isolated_config(
            llm_model="primary-503",
            llm_model2="fallback-ok",
            llm_url=base_url,
            llm_url2=base_url,
            llm_routing={
                "e2e_route": {
                    "primary": "llm1",
                    "fallback": True,
                    "fallback_target": "llm2",
                    "max_retries": 1,
                    "retry_delay_sec": 0,
                    "fallback_max_retries": 0,
                    "fallback_retry_delay_sec": 0,
                }
            },
        )
        monkeypatch.setattr(llm_service, "_current_config", cfg)
        result = await llm_service.callLLMTask(
            "e2e_route",
            [{"role": "user", "content": "retry and fallback"}],
        )

    assert result == "fallback-ok"
    assert request_models == ["primary-503", "primary-503", "fallback-ok"]


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 401, 429, 500, 503])
async def test_http_failure_body_is_visible_in_final_failure(
    monkeypatch,
    status: int,
):
    marker = f"provider-visible-{status}"

    async def completion(_request: web.Request) -> web.Response:
        return web.json_response(
            {"error": {"message": marker}},
            status=status,
        )

    provider = web.Application()
    provider.router.add_post("/v1/chat/completions", completion)
    async with _running_app(provider) as base_url:
        cfg = _isolated_config(
            llm_model=f"http-{status}",
            llm_url=base_url,
            llm_routing={
                "e2e_failure": {
                    "primary": "llm1",
                    "fallback": False,
                    "max_retries": 0,
                    "retry_delay_sec": 0,
                    "fallback_max_retries": 0,
                    "fallback_retry_delay_sec": 0,
                }
            },
        )
        monkeypatch.setattr(llm_service, "_current_config", cfg)
        result = await llm_service.callLLMTask(
            "e2e_failure",
            [{"role": "user", "content": "surface provider error"}],
        )

    assert result.startswith("[LLM 실패]")
    assert str(status) in result
    assert marker in result


@pytest.mark.asyncio
async def test_blank_success_body_is_retried_and_becomes_explicit_failure(monkeypatch):
    call_count = 0

    async def completion(_request: web.Request) -> web.Response:
        nonlocal call_count
        call_count += 1
        return web.json_response(
            {"choices": [{"message": {"content": " \n\t"}}]}
        )

    provider = web.Application()
    provider.router.add_post("/v1/chat/completions", completion)
    async with _running_app(provider) as base_url:
        cfg = _isolated_config(
            llm_url=base_url,
            llm_routing={
                "e2e_blank": {
                    "primary": "llm1",
                    "fallback": False,
                    "max_retries": 2,
                    "retry_delay_sec": 0,
                    "fallback_max_retries": 0,
                    "fallback_retry_delay_sec": 0,
                }
            },
        )
        monkeypatch.setattr(llm_service, "_current_config", cfg)
        result = await llm_service.callLLMTask(
            "e2e_blank",
            [{"role": "user", "content": "blank response"}],
        )

    assert call_count == 3
    assert result.startswith("[LLM 실패]")
    assert "응답이 비어 있음" in result


@pytest.mark.asyncio
async def test_llm_test_sse_endpoint_dispatches_all_five_targets(monkeypatch):
    cfg = _isolated_config()
    monkeypatch.setattr(llm_service, "get_config", lambda: cfg.copy())

    async def tracked_stream(
        messages,
        *,
        slot,
        stream_observer,
        metadata_sink,
        json_mode=False,
        **_kwargs,
    ):
        assert messages == [{"role": "user", "content": "endpoint smoke"}]
        assert json_mode is True
        slot_number = int(slot[-1])
        stream_id = f"tracked-{slot}"
        await stream_observer({
            "type": "stream_open",
            "stream_id": stream_id,
            "llm_slot": slot,
        })
        await stream_observer({
            "type": "delta",
            "stream_id": stream_id,
            "llm_slot": slot,
            "text": f"stream:{slot_number}",
            "elapsed": 0.01,
            "ttft": 0.01,
        })
        metadata_sink.update({
            "completion_tokens": 1,
            "prompt_tokens": 1,
            "elapsed": 0.02,
            "tps": 50.0,
            "ttft": 0.01,
        })
        return f"stream:{slot_number}"

    monkeypatch.setattr(
        llm_service,
        "callLLMTrackedStream",
        tracked_stream,
    )

    for slot in range(1, llm_service.LLM_SLOT_COUNT + 1):
        suffix = "" if slot == 1 else str(slot)

        async def single(
            messages,
            model=None,
            json_mode=False,
            _slot=slot,
            **_kwargs,
        ):
            assert messages == [{"role": "user", "content": "endpoint smoke"}]
            return f"single:{_slot}:json={json_mode}"

        monkeypatch.setattr(llm_service, f"callLLM{suffix}", single)
        monkeypatch.setattr(llm_service, f"callLLMVision{suffix}", single)

    app = web.Application()
    app.router.add_post("/api/llm/test_stream", server.handle_api_llm_test_stream)
    async with _running_app(app) as base_url, ClientSession() as client:
        for slot in range(1, llm_service.LLM_SLOT_COUNT + 1):
            for use_stream in (False, True):
                response = await client.post(
                    f"{base_url}/api/llm/test_stream",
                    json={
                        "messages": [
                            {"role": "user", "content": "endpoint smoke"}
                        ],
                        "target": f"llm{slot}",
                        "stream": use_stream,
                        "json_mode": True,
                    },
                )
                assert response.status == 200
                events = _parse_sse(await response.text())
                event_types = [event_type for event_type, _data in events]
                assert event_types[-1] == "done"
                done = events[-1][1]
                expected = (
                    f"stream:{slot}"
                    if use_stream
                    else f"single:{slot}:json=True"
                )
                assert done["text"] == expected


@pytest.mark.asyncio
async def test_llm_test_endpoint_rejects_unknown_target():
    app = web.Application()
    app.router.add_post("/api/llm/test_stream", server.handle_api_llm_test_stream)
    async with _running_app(app) as base_url, ClientSession() as client:
        response = await client.post(
            f"{base_url}/api/llm/test_stream",
            json={
                "messages": [{"role": "user", "content": "invalid target"}],
                "target": "llm999",
            },
        )

    assert response.status == 400
    payload = await response.json()
    assert "llm999" not in payload["error"]
    assert "llm1" in payload["error"]
    assert server.queue_manager.added == []


@pytest.mark.asyncio
async def test_llm_test_endpoint_surfaces_stream_and_single_failures(monkeypatch):
    cfg = _isolated_config()
    history_records: list[dict] = []
    monkeypatch.setattr(llm_service, "get_config", lambda: cfg.copy())
    monkeypatch.setattr(
        lighbd_service,
        "_log_lighbd_history",
        history_records.append,
    )

    async def failed_single(*_args, **_kwargs):
        return "[LLM 실패] provider single body"

    async def failed_stream(*_args, **_kwargs):
        return "[LLM 실패] provider stream body"

    monkeypatch.setattr(llm_service, "callLLM", failed_single)
    monkeypatch.setattr(llm_service, "callLLMTrackedStream", failed_stream)

    app = web.Application()
    app.router.add_post("/api/llm/test_stream", server.handle_api_llm_test_stream)
    async with _running_app(app) as base_url, ClientSession() as client:
        single_response = await client.post(
            f"{base_url}/api/llm/test_stream",
            json={
                "messages": [{"role": "user", "content": "fail"}],
                "target": "llm1",
                "stream": False,
            },
        )
        # 본문은 반드시 블록 안에서 읽는다. SSE 응답은 완전히 버퍼링되지
        # 않아, 서버와 ClientSession 이 닫힌 뒤에 읽으면
        # "Connection closed" 로 간헐 실패한다.
        single_body = await single_response.text()
        stream_response = await client.post(
            f"{base_url}/api/llm/test_stream",
            json={
                "messages": [{"role": "user", "content": "fail"}],
                "target": "llm1",
                "stream": True,
            },
        )
        stream_body = await stream_response.text()

    single_events = _parse_sse(single_body)
    stream_events = _parse_sse(stream_body)
    assert single_events[-1] == (
        "error",
        {"error": "[LLM 실패] provider single body"},
    )
    assert stream_events[-1] == (
        "error",
        {"error": "[LLM 실패] provider stream body"},
    )
    assert [record["status"] for record in history_records] == [
        "error",
        "error",
    ]
    assert "provider single body" in history_records[0]["error"]
    assert "provider single body" in history_records[0]["output"]
    assert "provider stream body" in history_records[1]["error"]
    assert "provider stream body" in history_records[1]["output"]


@pytest.mark.asyncio
async def test_llm_test_endpoint_registers_queue_item(monkeypatch):
    async def stream(*_args, **_kwargs):
        return "ok"

    queue_probe = server.queue_manager
    cfg = _isolated_config()
    monkeypatch.setattr(llm_service, "callLLMTrackedStream", stream)
    monkeypatch.setattr(
        llm_service,
        "get_config",
        lambda: cfg.copy(),
    )

    app = web.Application()
    app.router.add_post("/api/llm/test_stream", server.handle_api_llm_test_stream)
    async with _running_app(app) as base_url, ClientSession() as client:
        response = await client.post(
            f"{base_url}/api/llm/test_stream",
            json={
                "messages": [{"role": "user", "content": "queue"}],
                "target": "llm1",
                "stream": True,
            },
        )
        await response.read()

    assert queue_probe.added
    assert queue_probe.added[0]["item_type"] == "llm_test"
    assert queue_probe.added[0]["item_type"] in queue_manager_module.LLM_TYPES
    assert callable(queue_probe.added[0]["runtime_handler"])
    assert queue_probe.added[0]["params"]["execution_id"]


@pytest.mark.asyncio
async def test_llm_test_endpoint_records_lb_detail(monkeypatch):
    history_records: list[dict] = []
    cfg = _isolated_config()

    async def stream(*_args, **_kwargs):
        return "history-ok"

    monkeypatch.setattr(llm_service, "callLLMTrackedStream", stream)
    monkeypatch.setattr(
        llm_service,
        "get_config",
        lambda: cfg.copy(),
    )
    monkeypatch.setattr(
        lighbd_service,
        "_log_lighbd_history",
        history_records.append,
    )

    app = web.Application()
    app.router.add_post("/api/llm/test_stream", server.handle_api_llm_test_stream)
    async with _running_app(app) as base_url, ClientSession() as client:
        response = await client.post(
            f"{base_url}/api/llm/test_stream",
            json={
                "messages": [{"role": "user", "content": "history"}],
                "target": "llm1",
                "stream": True,
            },
        )
        await response.read()

    assert history_records
    assert history_records[-1]["output"] == "history-ok"
    assert history_records[-1]["execution_id"]
    assert history_records[-1]["history_id"] == history_records[-1]["execution_id"]


@pytest.mark.asyncio
async def test_llm_test_live_stream_has_id_and_can_be_stopped(monkeypatch):
    cfg = _isolated_config(llm_stream=False, llm_max_concurrency=2)
    monkeypatch.setattr(llm_service, "_current_config", cfg)
    frontend_events: list[dict] = []

    async def controlled_stream(messages, service, model):
        assert messages == [{"role": "user", "content": "stop me"}]
        yield {"type": "start", "service": service, "model": model}
        yield {
            "type": "delta",
            "text": "partial",
            "elapsed": 0.01,
            "ttft": 0.01,
        }
        await asyncio.Future()

    async def capture_stream(event):
        frontend_events.append(dict(event))

    monkeypatch.setattr(
        llm_service,
        "_dispatch_stream",
        controlled_stream,
    )
    monkeypatch.setattr(
        llm_service,
        "_stream_notify_func",
        capture_stream,
    )

    app = web.Application()
    app.router.add_post(
        "/api/llm/test_stream",
        server.handle_api_llm_test_stream,
    )
    app.router.add_post(
        "/api/llm/streams/{stream_id}/control",
        server.handle_api_llm_stream_control,
    )
    async with _running_app(app) as base_url, ClientSession() as client:
        response = await client.post(
            f"{base_url}/api/llm/test_stream",
            json={
                "messages": [{"role": "user", "content": "stop me"}],
                "target": "llm1",
                "stream": True,
            },
        )
        snapshot = None
        for _ in range(200):
            streams = llm_service.get_active_streams()
            if streams and streams[0].get("text") == "partial":
                snapshot = streams[0]
                break
            await asyncio.sleep(0)
        assert snapshot is not None
        assert snapshot["stream_id"]
        assert snapshot["task_key"] == "llm_test"
        assert snapshot["call_name"] == "LLM TEST"

        control_response = await client.post(
            (
                f"{base_url}/api/llm/streams/"
                f"{snapshot['stream_id']}/control"
            ),
            json={"action": "cancel"},
        )
        assert control_response.status == 200
        assert (await control_response.json())["success"] is True
        events = _parse_sse(await response.text())

    assert events[-1][0] == "cancelled"
    assert events[-1][1]["partial_text"] == "partial"
    assert llm_service.get_active_streams() == []
    assert frontend_events
    assert all(event.get("stream_id") for event in frontend_events)
    assert all(event.get("task_key") == "llm_test" for event in frontend_events)


@pytest.mark.asyncio
async def test_real_queue_dispatches_llm_test_without_serializing_handler():
    manager = queue_manager_module.QueueManager()
    item = queue_manager_module.QueueItem(
        id="llm-test-runtime",
        type="llm_test",
        label="LLM test runtime",
        params={"target": "llm1"},
    )

    async def runtime_handler(received):
        assert received is item
        return {"status": "ok"}

    item._runtime_handler = runtime_handler

    assert manager._item_execution_area(item) == ("llm", "llm")
    assert await manager._execute_item(item) == {"status": "ok"}
    assert "_runtime_handler" not in item.to_dict()


@pytest.mark.asyncio
async def test_validator_exhaustion_is_an_explicit_failure(monkeypatch):
    async def completion(_request: web.Request) -> web.Response:
        return web.json_response(
            {"choices": [{"message": {"content": "not-json"}}]}
        )

    provider = web.Application()
    provider.router.add_post("/v1/chat/completions", completion)
    async with _running_app(provider) as base_url:
        cfg = _isolated_config(
            llm_url=base_url,
            llm_routing={
                "e2e_invalid": {
                    "primary": "llm1",
                    "fallback": False,
                    "max_retries": 1,
                    "retry_delay_sec": 0,
                    "fallback_max_retries": 0,
                    "fallback_retry_delay_sec": 0,
                }
            },
        )
        monkeypatch.setattr(llm_service, "_current_config", cfg)
        result = await llm_service.callLLMTask(
            "e2e_invalid",
            [{"role": "user", "content": "must be json"}],
            result_validator=lambda value: (
                str(value).startswith("{"),
                "JSON 형식 아님",
            ),
        )

    assert result.startswith("[LLM 실패]")
    assert "JSON 형식 아님" in result


@pytest.mark.asyncio
async def test_fenced_json_recovery_composes_with_execution_result(monkeypatch):
    """복구 가능한 fenced JSON은 공통 실행 계층에서 실패/재시도로 오판되지 않는다."""
    raw = '설명\n```json\n{"ok": true, "value": 7}\n```\n끝'
    calls = 0

    async def routed_slot(
        _slot,
        _messages,
        model=None,
        json_mode=False,
    ):
        nonlocal calls
        calls += 1
        return raw

    cfg = _isolated_config(
        llm_stream=False,
        llm_routing={
            "json_recovery": {
                "primary": "llm1",
                "fallback": False,
                "max_retries": 1,
                "retry_delay_sec": 0,
            }
        },
    )
    monkeypatch.setattr(llm_service, "_current_config", cfg)
    monkeypatch.setattr(llm_service, "_call_routed_text_slot", routed_slot)

    result = await llm_service.callLLMTaskResult(
        "json_recovery",
        [{"role": "user", "content": "return JSON"}],
        json_mode=True,
        result_validator=lambda value: (
            illustration_context_pipeline._json_object_from_text(value)
            == {"ok": True, "value": 7},
            "JSON 복구 실패",
        ),
    )

    assert result.accepted is True
    assert result.text == raw
    assert result.raw_response == raw
    assert calls == 1
    assert [event["type"] for event in result.events] == [
        "attempt_start",
        "attempt_success",
        "execution_complete",
    ]


@pytest.mark.asyncio
async def test_pipeline_retry_attempt_failure_keeps_raw_response_in_lb_detail(
    monkeypatch,
):
    responses = iter(["RAW_INVALID_ATTEMPT", "VALID_ATTEMPT"])
    history_records: list[dict] = []

    async def routed_slot(
        _slot,
        _messages,
        model=None,
        json_mode=False,
    ):
        return next(responses)

    cfg = _isolated_config(
        llm_stream=False,
        llm_routing={
            "illustration_call1": {
                "primary": "llm1",
                "fallback": False,
                "max_retries": 1,
                "retry_delay_sec": 0,
                "fallback_max_retries": 0,
                "fallback_retry_delay_sec": 0,
            }
        },
    )
    monkeypatch.setattr(llm_service, "_current_config", cfg)
    monkeypatch.setattr(llm_service, "_call_routed_text_slot", routed_slot)
    monkeypatch.setattr(
        illustration_context_pipeline.lighbd_service,
        "_log_lighbd_history",
        history_records.append,
    )

    result = await illustration_context_pipeline._call_pipeline_llm(
        "CALL1",
        [{"role": "user", "content": "validate retry history"}],
        result_validator=lambda value: (
            value == "VALID_ATTEMPT",
            "synthetic validation failure",
        ),
    )

    assert result == "VALID_ATTEMPT"
    failure_records = [
        record
        for record in history_records
        if record.get("status") == "error"
    ]
    assert failure_records
    assert failure_records[0]["output"] == "RAW_INVALID_ATTEMPT"
    assert "synthetic validation failure" in failure_records[0]["error"]
    final_record = next(
        record for record in history_records if record.get("status") == "ok"
    )
    assert failure_records[0]["execution_id"] == final_record["execution_id"]
    assert failure_records[0]["parent_history_id"] == final_record["history_id"]
    assert failure_records[0]["attempt_id"].endswith(":primary:llm1:1")


def test_embedding_config_log_masks_api_key(monkeypatch):
    secret = "E2E_SECRET_MUST_NOT_APPEAR"
    isolated = dict(embedding_service._current_config)
    isolated["embedding_api_key"] = ""
    messages: list[str] = []
    monkeypatch.setattr(embedding_service, "_current_config", isolated)
    monkeypatch.setattr(embedding_service, "_log", messages.append)

    embedding_service.update_config({"embedding_api_key": secret})

    assert messages
    assert all(secret not in message for message in messages)
    assert any("api_key=set" in message for message in messages)


def test_api_key_inputs_remain_visible_by_product_design():
    source = (
        Path(__file__).resolve().parents[1] / "frontend" / "index.html"
    ).read_text(encoding="utf-8")
    input_ids = [
        "setting-llm-api-key",
        "setting-llm-api-key2",
        "setting-llm-api-key3",
        "setting-llm-api-key4",
        "setting-llm-api-key5",
        "setting-embedding-api-key",
        "setting-chansub-api-key",
        "setting-chansub-rotation-api-key",
    ]
    for input_id in input_ids:
        match = re.search(
            rf"<input\b[^>]*\bid=[\"']{re.escape(input_id)}[\"'][^>]*>",
            source,
            re.IGNORECASE,
        )
        assert match, f"API 키 input을 찾지 못함: {input_id}"
        assert re.search(
            r"\btype=[\"']text[\"']",
            match.group(0),
            re.IGNORECASE,
        ), f"API 키 input이 식별 가능한 text 타입이 아님: {input_id}"


@pytest.mark.asyncio
async def test_queue_websocket_logging_is_cp949_safe(monkeypatch):
    class FakeWebSocket:
        closed = False
        _req = None

        async def send_json(self, _message):
            return None

    monkeypatch.setattr(
        server,
        "frontend_ws_connections",
        {
            "cp949-client": {
                "ws": FakeWebSocket(),
                "last_pong": time.time(),
            }
        },
    )
    raw = io.BytesIO()
    console = io.TextIOWrapper(raw, encoding="cp949")
    with redirect_stdout(console):
        await _REAL_NOTIFY_FRONTEND("queue_updated", {"items": []})
        console.flush()

    output = raw.getvalue().decode("cp949")
    assert "[SEND]" in output
    assert "[OK]" in output


@pytest.mark.asyncio
async def test_slow_retry_duplicate_still_honors_global_route_retry(monkeypatch):
    attempt_names: list[str] = []
    duplicate_calls = 0
    primary_cancelled = asyncio.Event()
    history_records: list[dict] = []
    history_updates: list[dict] = []

    async def routed_slot(
        _slot,
        messages,
        model=None,
        json_mode=False,
    ):
        nonlocal duplicate_calls
        metadata = dict(llm_service._stream_metadata_ctx.get() or {})
        call_name = str(metadata.get("call_name") or "")
        attempt_names.append(call_name)
        protected_tokens = re.findall(
            illustration_context_pipeline._PROTECTED_SLOT_TOKEN_RE,
            str(messages[-1].get("content") or ""),
        )
        token = protected_tokens[0]
        if "1/2" in call_name:
            return f"Fast translation.\n\n{token}"
        if "느리다고? 다시해!" in call_name:
            duplicate_calls += 1
            if duplicate_calls == 1:
                return "invalid duplicate without slot marker"
            return f"Duplicate retry wins.\n\n{token}"
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            primary_cancelled.set()
            raise

    cfg = _isolated_config(
        llm_stream=False,
        llm_routing={
            "illustration_call1_backtranslate": {
                "primary": "llm1",
                "fallback": False,
                "max_retries": 1,
                "retry_delay_sec": 0,
                "fallback_max_retries": 0,
                "fallback_retry_delay_sec": 0,
            }
        },
    )
    monkeypatch.setattr(llm_service, "_current_config", cfg)
    monkeypatch.setattr(llm_service, "_call_routed_text_slot", routed_slot)
    monkeypatch.setattr(
        illustration_context_pipeline.lighbd_service,
        "_log_lighbd_history",
        history_records.append,
    )
    monkeypatch.setattr(
        illustration_context_pipeline.lighbd_service,
        "_update_lighbd_history_records",
        lambda updates: history_updates.append(updates) or len(updates),
    )

    translated, statuses = (
        await illustration_context_pipeline.backtranslate_current_context(
            "빠른 문장.\n\n[Slot 0]\n\n느린 문장.\n\n[Slot 1]",
            "Translate while preserving protected slot markers.",
            "Hana",
            2,
            failure_strategy="retry_abort",
            slow_retry_enabled=True,
            slow_retry_remaining=1,
            slow_retry_progress_enabled=True,
            slow_retry_progress_threshold=50,
            slow_retry_tps_enabled=False,
            slow_retry_condition_operator="and",
        )
    )

    assert "Duplicate retry wins." in translated
    assert statuses[1]["winner"] == "duplicate"
    assert duplicate_calls == 2
    assert primary_cancelled.is_set()
    assert len(attempt_names) == 4
    assert history_records
    assert history_updates
    linked_updates = [
        update
        for batch in history_updates
        for update in batch.values()
        if update.get("parent_execution_id")
    ]
    assert len(linked_updates) >= 2
    assert len({
        update["parent_execution_id"]
        for update in linked_updates
    }) == 1


@pytest.mark.asyncio
async def test_call_llm_task_force_slot_bypasses_routing_to_one_call(monkeypatch):
    """force_slot 이 정해지면 primary→fallback 라우팅/재시도를 건너뛰고 해당 슬롯 1회만 호출.

    CALL2-DETAIL 의 ①전부예측(primary)/②실패분만(fallback) 교대 루프가 단계별로
    지정 슬롯을 1회씩만 부르도록 쓰는 force_slot 계약을 고정한다."""
    cfg = _isolated_config(llm_routing={
        "force_task": {
            "primary": "llm1",
            "fallback_target": "llm2",
            "max_retries": 3,
            "retry_delay_sec": 0,
            "fallback_max_retries": 3,
            "fallback_retry_delay_sec": 0,
        },
    })
    monkeypatch.setattr(llm_service, "_current_config", cfg)

    invoked_slots: list[str] = []

    async def fake_slot(slot, messages, model=None, json_mode=False):
        invoked_slots.append(slot)
        return "force-ok"

    monkeypatch.setattr(llm_service, "_call_routed_text_slot", fake_slot)

    result = await llm_service.callLLMTaskResult(
        "force_task",
        [{"role": "user", "content": "hi"}],
        force_slot="llm2",
    )

    assert result.accepted is True
    assert result.final_slot == "llm2"
    assert result.final_phase == "forced"
    # primary(llm1)·fallback 자동 전환 모두 일어나지 않고 지정 슬롯 1회만.
    assert invoked_slots == ["llm2"]


@pytest.mark.asyncio
async def test_call_llm_task_force_slot_invalid_falls_back_to_normal_routing(monkeypatch):
    """force_slot 이 슬롯 id가 아니면 무시하고 일반 primary→fallback 라우팅을 따른다."""
    cfg = _isolated_config(llm_routing={
        "force_task": {
            "primary": "llm1",
            "fallback_target": "llm2",
            "max_retries": 0,
            "retry_delay_sec": 0,
            "fallback_max_retries": 0,
            "fallback_retry_delay_sec": 0,
        },
    })
    monkeypatch.setattr(llm_service, "_current_config", cfg)

    invoked_slots: list[str] = []

    async def fake_slot(slot, messages, model=None, json_mode=False):
        invoked_slots.append(slot)
        return "routing-ok"

    monkeypatch.setattr(llm_service, "_call_routed_text_slot", fake_slot)

    result = await llm_service.callLLMTaskResult(
        "force_task",
        [{"role": "user", "content": "hi"}],
        force_slot="not-a-slot",
    )

    assert result.accepted is True
    assert result.final_slot == "llm1"
    assert result.final_phase == "primary"
    assert invoked_slots == ["llm1"]
