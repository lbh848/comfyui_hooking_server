import asyncio
import base64
import binascii
import os
import struct
import sys
import zlib
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import server
from modes import llm_pdf_prompt, llm_service


PDF_SERVICES = ("gemini", "vertex")


def _config(**overrides):
    values = llm_service.get_config()
    values.update(
        {
            "llm_service": "vertex",
            "llm_model": "gemini-test",
            "llm_pdf_prompt": True,
            "llm_gemini_base64": False,
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


def _sample_messages(include_image=True):
    user_content = [{"type": "text", "text": "암호명은 은하우체국입니다."}]
    if include_image:
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,AAAA"},
            }
        )
    return [
        {"role": "system", "content": "맥락 전체를 읽고 정확히 답하세요."},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": "기억했습니다."},
        {"role": "user", "content": "암호명만 답하세요."},
    ]


def _pdf_file_part(messages):
    return messages[-1]["content"][0]


def _solid_png_data_url(red: int, green: int, blue: int, size: int = 64) -> str:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", binascii.crc32(kind + payload) & 0xFFFFFFFF)
        )

    scanline = b"\x00" + bytes((red, green, blue)) * size
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(
            b"IHDR",
            struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0),
        )
        + chunk(b"IDAT", zlib.compress(scanline * size))
        + chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def test_all_llm_slots_register_independent_pdf_prompt_config():
    runtime = llm_service.get_config()
    for slot in range(1, llm_service.LLM_SLOT_COUNT + 1):
        suffix = "" if slot == 1 else str(slot)
        key = f"llm_pdf_prompt{suffix}"
        assert runtime[key] is False
        assert server.DEFAULT_CONFIG[key] is False


def test_one_point_pdf_packs_typical_long_context_into_one_page():
    assert llm_pdf_prompt.PDF_FONT_SIZE == 1.0
    assert len(llm_pdf_prompt._paginate_text("가" * 900_000)) == 1
    assert len(llm_pdf_prompt._paginate_text("가" * 1_000_000)) == 2

    pdf_bytes, page_count = llm_pdf_prompt.build_text_pdf(
        "첫 줄\n\n둘째 줄 😀 한글"
    )

    assert page_count == 1
    assert pdf_bytes.startswith(b"%PDF-1.7")
    assert pdf_bytes.rstrip().endswith(b"%%EOF")


def test_transcript_preserves_roles_and_moves_images_in_order():
    transcript, media_parts = llm_pdf_prompt.build_conversation_transcript(
        _sample_messages()
    )

    assert "--- SYSTEM 1 ---" in transcript
    assert "--- USER 2 ---" in transcript
    assert "--- ASSISTANT 3 ---" in transcript
    assert "은하우체국" in transcript
    assert "[Image 1 is attached separately" in transcript
    assert len(media_parts) == 1
    assert media_parts[0]["type"] == "image_url"


@pytest.mark.parametrize("encode_transcript", [False, True])
def test_pdf_message_transform_encodes_complete_transcript_and_keeps_image(
    monkeypatch, encode_transcript,
):
    captured = {}

    def fake_build_text_pdf(text):
        captured["pdf_text"] = text
        return b"%PDF-test", 1

    monkeypatch.setattr(llm_pdf_prompt, "build_text_pdf", fake_build_text_pdf)
    transformed, metadata = llm_pdf_prompt.prepare_pdf_prompt_messages(
        _sample_messages(),
        encode_transcript=encode_transcript,
    )

    if encode_transcript:
        decoded = base64.b64decode(captured["pdf_text"]).decode("utf-8")
        assert "은하우체국" in decoded
        assert "은하우체국" not in captured["pdf_text"]
        assert "Base64 block" in transformed[0]["content"]
    else:
        assert "은하우체국" in captured["pdf_text"]
        assert "complete role-labelled" in transformed[0]["content"]

    file_part = _pdf_file_part(transformed)
    assert file_part["type"] == "file"
    assert file_part["file"]["filename"] == llm_pdf_prompt.PDF_FILENAME
    assert file_part["file"]["file_data"].startswith(
        "data:application/pdf;base64,"
    )
    assert transformed[-1]["content"][1]["type"] == "image_url"
    assert metadata["image_count"] == 1
    assert metadata["base64_transcript"] is encode_transcript


def test_gemini_rest_body_contains_pdf_image_and_low_media_resolution(monkeypatch):
    monkeypatch.setattr(llm_service, "_current_config", _config())
    transformed, _ = llm_pdf_prompt.prepare_pdf_prompt_messages(
        _sample_messages(), encode_transcript=False
    )

    body = llm_service._build_gemini_request_body(
        transformed,
        "gemini-test",
    )
    parts = body["contents"][0]["parts"]

    assert parts[0]["inline_data"]["mime_type"] == "application/pdf"
    assert parts[1]["inline_data"]["mime_type"] == "image/png"
    assert body["generationConfig"]["mediaResolution"] == "MEDIA_RESOLUTION_LOW"


def test_vertex_sdk_parts_contain_pdf_and_separate_image():
    transformed, _ = llm_pdf_prompt.prepare_pdf_prompt_messages(
        _sample_messages(), encode_transcript=False
    )

    parts, system_instruction = llm_service._build_genai_contents(transformed)

    assert "complete role-labelled" in system_instruction
    assert len(parts) == 2
    assert parts[0].inline_data.mime_type == "application/pdf"
    assert parts[0].media_resolution.level == "MEDIA_RESOLUTION_LOW"
    assert parts[1].inline_data.mime_type == "image/png"


@pytest.mark.parametrize("service", PDF_SERVICES)
@pytest.mark.asyncio
async def test_pdf_and_base64_compose_before_native_dispatch(monkeypatch, service):
    monkeypatch.setattr(
        llm_service,
        "_current_config",
        _config(
            llm_service=service,
            llm_pdf_prompt=True,
            llm_gemini_base64=True,
        ),
    )
    seen = {}

    async def fake_dispatch_unlimited(messages, service, model):
        seen["messages"] = messages
        return base64.b64encode("은하우체국".encode("utf-8")).decode("ascii")

    monkeypatch.setattr(
        llm_service, "_dispatch_unlimited", fake_dispatch_unlimited
    )
    result = await llm_service._dispatch(
        _sample_messages(), service, "gemini-test"
    )

    assert result == "은하우체국"
    request_messages = seen["messages"]
    assert "PDF Base64 Data Transport Protocol" in request_messages[0]["content"]
    decoded_bootstrap = base64.b64decode(request_messages[1]["content"]).decode(
        "utf-8"
    )
    assert "PDF's embedded native text is one UTF-8 Base64 block" in decoded_bootstrap
    assert request_messages[2]["content"][0]["type"] == "file"
    assert request_messages[2]["content"][1]["type"] == "image_url"


@pytest.mark.asyncio
async def test_vertex_openai_does_not_apply_pdf_transform(monkeypatch):
    monkeypatch.setattr(
        llm_service,
        "_current_config",
        _config(
            llm_service="vertex-openai",
            llm_pdf_prompt=True,
            llm_gemini_base64=False,
        ),
    )
    original = _sample_messages()
    seen = {}

    async def fake_dispatch_unlimited(messages, service, model):
        seen["messages"] = messages
        return "ok"

    monkeypatch.setattr(
        llm_service, "_dispatch_unlimited", fake_dispatch_unlimited
    )
    result = await llm_service._dispatch(
        original, "vertex-openai", "gemini-test"
    )

    assert result == "ok"
    assert seen["messages"] is original


@pytest.mark.asyncio
async def test_stream_uses_pdf_and_base64_composition(monkeypatch):
    monkeypatch.setattr(
        llm_service,
        "_current_config",
        _config(llm_pdf_prompt=True, llm_gemini_base64=True),
    )
    expected = "스트림 성공"
    encoded = base64.b64encode(expected.encode("utf-8")).decode("ascii")

    async def fake_stream_unlimited(messages, service, model):
        assert messages[2]["content"][0]["type"] == "file"
        yield {"type": "start", "service": service, "model": model}
        yield {"type": "delta", "text": encoded}
        yield {"type": "done", "text": encoded}

    monkeypatch.setattr(
        llm_service, "_dispatch_stream_unlimited", fake_stream_unlimited
    )
    events = [
        event
        async for event in llm_service._dispatch_stream(
            _sample_messages(), "vertex", "gemini-test"
        )
    ]

    assert events[-1]["text"] == expected


@pytest.mark.asyncio
async def test_pdf_conversion_failure_is_logged_and_fails_closed(monkeypatch, capsys):
    monkeypatch.setattr(llm_service, "_current_config", _config())

    def fail_transform(messages, service):
        raise RuntimeError("pdf exploded")

    monkeypatch.setattr(
        llm_service, "_prepare_gemini_pdf_messages", fail_transform
    )
    result = await llm_service._dispatch(
        _sample_messages(), "vertex", "gemini-test"
    )

    assert result == "[LLM 실패] PDF 프롬프트 변환 오류: pdf exploded"
    captured = capsys.readouterr()
    assert "동기 요청 PDF 변환 실패" in captured.out
    assert "Traceback" in captured.err


def test_frontend_registers_pdf_control_for_every_llm_slot():
    html = (
        Path(__file__).resolve().parents[1] / "frontend" / "index.html"
    ).read_text(encoding="utf-8")

    for slot in range(1, llm_service.LLM_SLOT_COUNT + 1):
        suffix = "" if slot == 1 else str(slot)
        assert html.count(f'id="setting-llm-pdf-prompt{suffix}"') == 1
        assert html.count(f'id="llm-pdf-prompt{suffix}-row"') == 1
    assert "config[`llm_pdf_prompt${suffix}`]" in html
    assert "['gemini', 'vertex'].includes(meta.id)" in html


@pytest.mark.skipif(
    os.environ.get("RUN_LIVE_VERTEX_PDF_E2E") != "1",
    reason="실제 Vertex 자격증명을 사용하는 명시적 라이브 E2E",
)
@pytest.mark.parametrize("stream", [False, True], ids=["sync", "stream"])
@pytest.mark.asyncio
async def test_live_vertex_understands_one_point_pdf_transcript(
    monkeypatch, tmp_path, stream,
):
    key_path = llm_service._get_vertex_key_path()
    if not key_path:
        pytest.fail("key/vertex.json 서비스 계정이 없습니다")

    isolated_log_dir = tmp_path / "logs"
    monkeypatch.setattr(llm_service, "LOG_DIR", str(isolated_log_dir))
    monkeypatch.setattr(
        llm_service,
        "HISTORY_PATH",
        str(isolated_log_dir / "llm_history.jsonl"),
    )
    monkeypatch.setattr(
        llm_service,
        "_current_config",
        _config(
            llm_service="vertex",
            llm_model=os.environ.get(
                "VERTEX_PDF_E2E_MODEL", "gemini-3.7-flash"
            ),
                llm_pdf_prompt=True,
                llm_gemini_base64=False,
                llm_stream=stream,
                llm_temperature=0.1,
                llm_stream_idle_timeout_seconds=120,
        ),
    )
    monkeypatch.setattr(llm_service, "_stream_notify_func", None)
    expected = "EUNHA-585987"
    filler = "이 문장은 문서 이해 여부를 시험하는 주변 문맥입니다. " * 1500
    result = await llm_service.callLLM(
        [
            {
                "role": "system",
                "content": "Read the full conversation and follow the final request exactly.",
            },
            {
                "role": "user",
                "content": (
                    filler
                    + "The project code is EUNHA. The first number is 314159 and "
                    "the second number is 271828."
                ),
            },
            {"role": "assistant", "content": "I will remember those facts."},
            {
                "role": "user",
                "content": (
                    "Return only PROJECTCODE-HALFSUM, where HALFSUM is the sum of "
                    "the two numbers. Do not add punctuation or explanation."
                ),
            },
        ]
    )

    assert not result.startswith("[LLM 실패]"), result
    assert result.strip() == expected


@pytest.mark.skipif(
    os.environ.get("RUN_LIVE_VERTEX_PDF_E2E") != "1",
    reason="실제 Vertex 자격증명을 사용하는 명시적 라이브 E2E",
)
@pytest.mark.parametrize("stream", [False, True], ids=["sync", "stream"])
@pytest.mark.asyncio
async def test_live_vertex_understands_base64_pdf_and_separate_image(
    monkeypatch, tmp_path, stream,
):
    if not llm_service._get_vertex_key_path():
        pytest.fail("key/vertex.json 서비스 계정이 없습니다")

    isolated_log_dir = tmp_path / "logs"
    monkeypatch.setattr(llm_service, "LOG_DIR", str(isolated_log_dir))
    monkeypatch.setattr(
        llm_service,
        "HISTORY_PATH",
        str(isolated_log_dir / "llm_history.jsonl"),
    )
    monkeypatch.setattr(
        llm_service,
        "_current_config",
        _config(
            llm_service="vertex",
            llm_model=os.environ.get(
                "VERTEX_PDF_E2E_MODEL", "gemini-3.7-flash"
            ),
                llm_pdf_prompt=True,
                llm_gemini_base64=True,
                llm_stream=stream,
                llm_temperature=0.1,
                llm_stream_idle_timeout_seconds=120,
        ),
    )
    monkeypatch.setattr(llm_service, "_stream_notify_func", None)
    result = await llm_service.callLLM(
        [
            {"role": "system", "content": "Use the PDF and attached image together."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "The project code is EUNHA. Return only "
                            "PROJECTCODE-DOMINANTCOLOR for the separately attached image."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": _solid_png_data_url(0, 64, 255),
                        },
                    },
                ],
            },
        ]
    )

    assert not result.startswith("[LLM 실패]"), result
    assert result.strip().upper() == "EUNHA-BLUE"
