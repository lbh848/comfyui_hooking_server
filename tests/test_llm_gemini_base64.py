import base64
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import server
from modes import llm_service


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
    assert "Base64-Encoded Instruction Protocol" in encoded[0]["content"]
    assert encoded[1]["role"] == "system"
    assert base64.b64decode(encoded[1]["content"]).decode("utf-8") == "시스템 지시"
    assert encoded[2]["role"] == "user"
    assert (
        base64.b64decode(encoded[2]["content"][0]["text"]).decode("utf-8")
        == "장면을 분석해"
    )
    assert encoded[2]["content"][1] == image_part
    assert encoded[2]["content"][1] is not image_part


def test_gemini_base64_removes_conflicting_json_response_format(monkeypatch):
    monkeypatch.setattr(llm_service, "_current_config", _config())

    body = llm_service._build_gemini_request_body(
        [{"role": "user", "content": "응답"}],
        "gemini-test",
        custom_body=(
            '{"generationConfig": {'
            '"responseMimeType": "application/json", '
            '"responseSchema": {"type": "object"}'
            '}}'
        ),
    )

    generation_config = body["generationConfig"]
    assert "responseMimeType" not in generation_config
    assert "responseSchema" not in generation_config


@pytest.mark.asyncio
async def test_sync_gemini_base64_wraps_request_decodes_response_and_disables_json_mode(
    monkeypatch,
):
    monkeypatch.setattr(llm_service, "_current_config", _config())
    seen = {}
    plain_response = '{"scenes": [{"id": "한글"}]}'

    async def fake_dispatch_unlimited(messages, service, model):
        seen["messages"] = messages
        seen["service"] = service
        seen["model"] = model
        seen["response_format"] = llm_service._response_format_ctx.get()
        return base64.b64encode(plain_response.encode("utf-8")).decode("ascii")

    monkeypatch.setattr(
        llm_service, "_dispatch_unlimited", fake_dispatch_unlimited
    )
    outer_token = llm_service._response_format_ctx.set({"type": "json_object"})
    try:
        result = await llm_service._dispatch(
            [{"role": "user", "content": "JSON으로 답해"}],
            "gemini",
            "gemini-test",
        )
        assert llm_service._response_format_ctx.get() == {"type": "json_object"}
    finally:
        llm_service._response_format_ctx.reset(outer_token)

    assert result == plain_response
    assert seen["service"] == "gemini"
    assert seen["model"] == "gemini-test"
    assert seen["response_format"] is None
    assert "Base64-Encoded Instruction Protocol" in seen["messages"][0]["content"]
    assert (
        base64.b64decode(seen["messages"][1]["content"]).decode("utf-8")
        == "JSON으로 답해"
    )


@pytest.mark.asyncio
async def test_base64_toggle_does_not_modify_non_gemini_service(monkeypatch):
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


@pytest.mark.parametrize("slot", range(1, llm_service.LLM_SLOT_COUNT + 1))
@pytest.mark.asyncio
async def test_each_llm_slot_uses_its_own_base64_toggle_with_inherited_service(
    monkeypatch, slot,
):
    overrides = {
        "llm_service": "gemini",
        "llm_gemini_base64": slot == 1,
    }
    for configured_slot in range(2, llm_service.LLM_SLOT_COUNT + 1):
        overrides.update(
            {
                f"llm_service{configured_slot}": "",
                f"llm_model{configured_slot}": f"gemini-slot-{configured_slot}",
                f"llm_gemini_base64{configured_slot}": configured_slot == slot,
                f"llm_stream{configured_slot}": False,
            }
        )
    monkeypatch.setattr(llm_service, "_current_config", _config(**overrides))
    seen = {}

    async def fake_dispatch_unlimited(messages, service, model):
        seen["messages"] = messages
        seen["slot"] = llm_service._llm_slot_ctx.get()
        return base64.b64encode(f"slot {slot}".encode("utf-8")).decode("ascii")

    monkeypatch.setattr(
        llm_service, "_dispatch_unlimited", fake_dispatch_unlimited
    )
    call = getattr(llm_service, "callLLM" if slot == 1 else f"callLLM{slot}")
    result = await call([{"role": "user", "content": f"LLM{slot} 요청"}])

    assert result == f"slot {slot}"
    assert seen["slot"] == f"llm{slot}"
    assert "Base64-Encoded Instruction Protocol" in seen["messages"][0]["content"]


@pytest.mark.asyncio
async def test_gemini_base64_stream_emits_decoded_deltas_and_done(monkeypatch):
    monkeypatch.setattr(llm_service, "_current_config", _config())
    plain = "스트리밍 Base64 응답입니다."
    encoded = base64.b64encode(plain.encode("utf-8")).decode("ascii")

    async def fake_stream_unlimited(messages, service, model):
        assert "Base64-Encoded Instruction Protocol" in messages[0]["content"]
        yield {"type": "start", "service": service, "model": model}
        for start in range(0, len(encoded), 5):
            yield {"type": "delta", "text": encoded[start:start + 5]}
        yield {"type": "done", "text": encoded, "completion_tokens": 10}

    monkeypatch.setattr(
        llm_service, "_dispatch_stream_unlimited", fake_stream_unlimited
    )
    events = [
        event
        async for event in llm_service._dispatch_stream(
            [{"role": "user", "content": "요청"}],
            "gemini",
            "gemini-test",
        )
    ]

    decoded_deltas = "".join(
        event.get("text", "") for event in events if event["type"] == "delta"
    )
    assert plain.startswith(decoded_deltas)
    assert events[-1]["type"] == "done"
    assert events[-1]["text"] == plain


def test_frontend_registers_base64_control_for_every_llm_slot():
    html = (
        Path(__file__).resolve().parents[1] / "frontend" / "index.html"
    ).read_text(encoding="utf-8")

    for slot in range(1, llm_service.LLM_SLOT_COUNT + 1):
        suffix = "" if slot == 1 else str(slot)
        assert html.count(f'id="setting-llm-gemini-base64{suffix}"') == 1
        assert html.count(f'id="llm-gemini-base64{suffix}-row"') == 1
    assert "config[`llm_gemini_base64${suffix}`]" in html
    assert "meta.id === 'gemini'" in html
