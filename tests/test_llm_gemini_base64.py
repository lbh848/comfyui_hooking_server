import asyncio
import base64
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import server
from modes import llm_service


BASE64_TRANSPORT_SERVICES = ("gemini", "vertex", "vertex-openai")


def _config(**overrides):
    values = llm_service.get_config()
    values.update(
        {
            "llm_service": "gemini",
            "llm_model": "gemini-test",
            "llm_gemini_base64": True,
            "llm_stream": False,
            "llm_max_concurrency": 1,
        }
    )
    values.update(overrides)
    return llm_service._ContextConfig(values)


@pytest.fixture(autouse=True)
def _clear_request_gates():
    llm_service._request_gates_by_loop.clear()
    yield
    llm_service._request_gates_by_loop.clear()


def test_all_llm_slots_register_independent_gemini_base64_config():
    runtime = llm_service.get_config()
    for slot in range(1, llm_service.LLM_SLOT_COUNT + 1):
        suffix = "" if slot == 1 else str(slot)
        key = f"llm_gemini_base64{suffix}"
        assert key in runtime
        assert runtime[key] is False
        assert key in server.DEFAULT_CONFIG
        assert server.DEFAULT_CONFIG[key] is False


def test_gemini_base64_encoder_preserves_roles_and_image_parts():
    image_part = {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,AAAA"},
    }
    messages = [
        {"role": "system", "content": "시스템 지시"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "장면을 분석해"},
                image_part,
            ],
        },
    ]

    encoded = llm_service._encode_gemini_base64_messages(messages)

    assert encoded[0]["role"] == "system"
    assert "Base64-Encoded Input Protocol" in encoded[0]["content"]
    assert "Do not Base64-encode the response" in encoded[0]["content"]
    assert encoded[1]["role"] == "system"
    assert base64.b64decode(encoded[1]["content"]).decode("utf-8") == "시스템 지시"
    assert encoded[2]["role"] == "user"
    assert (
        base64.b64decode(encoded[2]["content"][0]["text"]).decode("utf-8")
        == "장면을 분석해"
    )
    assert encoded[2]["content"][1] == image_part
    assert encoded[2]["content"][1] is not image_part


def test_gemini_base64_input_keeps_json_response_format(monkeypatch):
    monkeypatch.setattr(llm_service, "_current_config", _config())
    token = llm_service._response_format_ctx.set({"type": "json_object"})
    try:
        body = llm_service._build_gemini_request_body(
            [{"role": "user", "content": "응답"}],
            "gemini-test",
        )
    finally:
        llm_service._response_format_ctx.reset(token)

    generation_config = body["generationConfig"]
    assert generation_config["responseMimeType"] == "application/json"
    assert "responseSchema" not in generation_config


@pytest.mark.parametrize("service", BASE64_TRANSPORT_SERVICES)
@pytest.mark.asyncio
async def test_sync_base64_transport_wraps_request_and_keeps_plain_response(
    monkeypatch, service,
):
    monkeypatch.setattr(
        llm_service,
        "_current_config",
        _config(llm_service=service, llm_model=f"{service}-test"),
    )
    seen = {}
    plain_response = '{"scenes": [{"id": "한글"}]}'

    async def fake_dispatch_unlimited(messages, service, model):
        seen["messages"] = messages
        seen["service"] = service
        seen["model"] = model
        seen["response_format"] = llm_service._response_format_ctx.get()
        return plain_response

    monkeypatch.setattr(
        llm_service, "_dispatch_unlimited", fake_dispatch_unlimited
    )
    outer_token = llm_service._response_format_ctx.set({"type": "json_object"})
    try:
        result = await llm_service._dispatch(
            [{"role": "user", "content": "JSON으로 답해"}],
            service,
            f"{service}-test",
        )
        assert llm_service._response_format_ctx.get() == {"type": "json_object"}
    finally:
        llm_service._response_format_ctx.reset(outer_token)

    assert result == plain_response
    assert seen["service"] == service
    assert seen["model"] == f"{service}-test"
    assert seen["response_format"] == {"type": "json_object"}
    assert "Base64-Encoded Input Protocol" in seen["messages"][0]["content"]
    assert (
        base64.b64decode(seen["messages"][1]["content"]).decode("utf-8")
        == "JSON으로 답해"
    )


@pytest.mark.parametrize("service", BASE64_TRANSPORT_SERVICES)
@pytest.mark.asyncio
async def test_base64_looking_plain_response_is_not_decoded(monkeypatch, service):
    monkeypatch.setattr(
        llm_service,
        "_current_config",
        _config(llm_service=service, llm_model=f"{service}-test"),
    )

    async def fake_dispatch_unlimited(messages, service, model):
        return "YWJj"

    monkeypatch.setattr(
        llm_service, "_dispatch_unlimited", fake_dispatch_unlimited
    )

    result = await llm_service._dispatch(
        [{"role": "user", "content": "원문 그대로 답해"}],
        service,
        f"{service}-test",
    )

    assert result == "YWJj"


@pytest.mark.asyncio
async def test_base64_toggle_does_not_modify_unsupported_service(monkeypatch):
    monkeypatch.setattr(
        llm_service,
        "_current_config",
        _config(llm_service="openai", llm_gemini_base64=True),
    )
    original_messages = [{"role": "user", "content": "그대로"}]
    seen = {}

    async def fake_dispatch_unlimited(messages, service, model):
        seen["messages"] = messages
        return "plain response"

    monkeypatch.setattr(
        llm_service, "_dispatch_unlimited", fake_dispatch_unlimited
    )
    result = await llm_service._dispatch(
        original_messages, "openai", "openai-test"
    )

    assert result == "plain response"
    assert seen["messages"] is original_messages


@pytest.mark.parametrize("service", BASE64_TRANSPORT_SERVICES)
@pytest.mark.parametrize("slot", range(1, llm_service.LLM_SLOT_COUNT + 1))
@pytest.mark.asyncio
async def test_each_llm_slot_uses_its_own_base64_toggle_with_inherited_service(
    monkeypatch, slot, service,
):
    overrides = {
        "llm_service": service,
        "llm_gemini_base64": slot == 1,
    }
    for configured_slot in range(2, llm_service.LLM_SLOT_COUNT + 1):
        overrides.update(
            {
                f"llm_service{configured_slot}": "",
                f"llm_model{configured_slot}": f"{service}-slot-{configured_slot}",
                f"llm_gemini_base64{configured_slot}": configured_slot == slot,
                f"llm_stream{configured_slot}": False,
            }
        )
    monkeypatch.setattr(llm_service, "_current_config", _config(**overrides))
    seen = {}

    async def fake_dispatch_unlimited(messages, service, model):
        seen["messages"] = messages
        seen["slot"] = llm_service._llm_slot_ctx.get()
        return f"slot {slot}"

    monkeypatch.setattr(
        llm_service, "_dispatch_unlimited", fake_dispatch_unlimited
    )
    call = getattr(llm_service, "callLLM" if slot == 1 else f"callLLM{slot}")
    result = await call([{"role": "user", "content": f"LLM{slot} 요청"}])

    assert result == f"slot {slot}"
    assert seen["slot"] == f"llm{slot}"
    assert "Base64-Encoded Input Protocol" in seen["messages"][0]["content"]


@pytest.mark.parametrize("service", BASE64_TRANSPORT_SERVICES)
@pytest.mark.asyncio
async def test_base64_transport_stream_passes_plain_deltas_and_done(
    monkeypatch, service,
):
    monkeypatch.setattr(
        llm_service,
        "_current_config",
        _config(llm_service=service, llm_model=f"{service}-test"),
    )
    plain = "스트리밍 Base64 응답입니다."
    async def fake_stream_unlimited(messages, service, model):
        assert "Base64-Encoded Input Protocol" in messages[0]["content"]
        yield {"type": "start", "service": service, "model": model}
        for start in range(0, len(plain), 5):
            yield {"type": "delta", "text": plain[start:start + 5]}
        yield {"type": "done", "text": plain, "completion_tokens": 10}

    monkeypatch.setattr(
        llm_service, "_dispatch_stream_unlimited", fake_stream_unlimited
    )
    events = [
        event
        async for event in llm_service._dispatch_stream(
            [{"role": "user", "content": "요청"}],
            service,
            f"{service}-test",
        )
    ]

    plain_deltas = "".join(
        event.get("text", "") for event in events if event["type"] == "delta"
    )
    assert plain_deltas == plain
    assert events[-1]["type"] == "done"
    assert events[-1]["text"] == plain


@pytest.mark.asyncio
async def test_base64_stream_can_be_closed_from_a_different_async_context(monkeypatch):
    monkeypatch.setattr(llm_service, "_current_config", _config())

    async def fake_stream_unlimited(messages, service, model):
        assert llm_service._response_format_ctx.get() == {"type": "json_object"}
        yield {"type": "start", "service": service, "model": model}
        yield {"type": "done", "text": "ok"}

    monkeypatch.setattr(
        llm_service,
        "_dispatch_stream_unlimited",
        fake_stream_unlimited,
    )
    outer_token = llm_service._response_format_ctx.set({"type": "json_object"})
    stream = llm_service._dispatch_stream(
        [{"role": "user", "content": "요청"}],
        "vertex",
        "vertex-test",
    )
    try:
        first_event = await asyncio.create_task(anext(stream))
        assert first_event["type"] == "start"
        assert llm_service._response_format_ctx.get() == {"type": "json_object"}
        await stream.aclose()
        assert llm_service._response_format_ctx.get() == {"type": "json_object"}
    finally:
        llm_service._response_format_ctx.reset(outer_token)


def test_frontend_registers_base64_control_for_every_llm_slot():
    html = (
        Path(__file__).resolve().parents[1] / "frontend" / "index.html"
    ).read_text(encoding="utf-8")

    for slot in range(1, llm_service.LLM_SLOT_COUNT + 1):
        suffix = "" if slot == 1 else str(slot)
        assert html.count(f'id="setting-llm-gemini-base64{suffix}"') == 1
        assert html.count(f'id="llm-gemini-base64{suffix}-row"') == 1
    assert "config[`llm_gemini_base64${suffix}`]" in html
    assert "['gemini', 'vertex', 'vertex-openai'].includes(meta.id)" in html
    assert "텍스트 요청만 UTF-8 Base64로 감쌉니다" in html
    assert "응답은 안정적인 파싱을 위해 원문 UTF-8로 받으며" in html


@pytest.mark.skipif(
    os.environ.get("RUN_LIVE_VERTEX_BASE64_E2E") != "1",
    reason="실제 Vertex 자격증명을 사용하는 명시적 라이브 E2E",
)
@pytest.mark.parametrize("stream", [False, True], ids=["sync", "stream"])
@pytest.mark.parametrize("service", ["vertex", "vertex-openai"])
@pytest.mark.asyncio
async def test_live_vertex_base64_round_trip(monkeypatch, tmp_path, service, stream):
    key_path = llm_service._get_vertex_key_path()
    if not key_path:
        pytest.fail("key/vertex.json 또는 사용할 수 있는 Vertex 서비스 계정 JSON이 없습니다")

    if service == "vertex":
        model = os.environ.get("VERTEX_NATIVE_E2E_MODEL", "gemini-2.5-flash")
        url = ""
    else:
        model = os.environ.get(
            "VERTEX_OPENAI_E2E_MODEL",
            "google/gemini-2.5-flash",
        )
        url = os.environ.get("VERTEX_OPENAI_E2E_LOCATION", "us-central1")

    isolated_log_dir = tmp_path / "logs"
    monkeypatch.setattr(llm_service, "LOG_DIR", str(isolated_log_dir))
    monkeypatch.setattr(
        llm_service,
        "HISTORY_PATH",
        str(isolated_log_dir / "llm_history.jsonl"),
    )
    monkeypatch.setattr(
        llm_service,
        "HISTORY_BACKUP_DIR",
        str(tmp_path / "backups"),
    )
    monkeypatch.setattr(
        llm_service,
        "HISTORY_BACKUP_PATH",
        str(tmp_path / "backups" / "llm_history.jsonl.bak"),
    )
    monkeypatch.setattr(
        llm_service,
        "_current_config",
        _config(
            llm_service=service,
            llm_model=model,
            llm_url=url,
            llm_gemini_base64=True,
            llm_stream=stream,
            llm_stream_idle_timeout_seconds=120,
        ),
    )
    monkeypatch.setattr(llm_service, "_stream_notify_func", None)

    expected_answer = "585987"
    result = await llm_service.callLLM(
        [
            {
                "role": "user",
                "content": (
                    "Calculate 314159 + 271828. "
                    "Return only the six ASCII digits of the answer, with no punctuation."
                ),
            }
        ]
    )

    assert not result.startswith("[LLM 실패]"), result
    assert result.strip() == expected_answer
