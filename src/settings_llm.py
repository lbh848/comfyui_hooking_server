"""Run settings diagnostics through the existing LLM engine."""
from __future__ import annotations

import asyncio
import base64
import binascii
import io
import time
from typing import Annotated, Literal

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, model_validator
from llm_settings import LLM_SP, LLM_SP_SLOTS, SLOT_PATTERN


class TestImage(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    b64: Annotated[str, Field(min_length=1, max_length=12 * 1024 * 1024)]
    mime: Literal["image/png", "image/jpeg", "image/webp"]

    @model_validator(mode="after")
    def validate_image(self):
        try:
            data = base64.b64decode(self.b64, validate=True)
            with Image.open(io.BytesIO(data)) as image:
                if image.width * image.height > 40_000_000 or Image.MIME.get(image.format) != self.mime:
                    raise ValueError("Invalid image")
                image.verify()
        except (ValueError, OSError, binascii.Error, Image.DecompressionBombError):
            raise ValueError("Invalid test image") from None
        return self


class LlmTestRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    target: Annotated[str, Field(pattern=rf"^{SLOT_PATTERN}$")]
    prompt: Annotated[str, Field(min_length=1, max_length=32768)]
    images: Annotated[list[TestImage], Field(max_length=6)] = []


class LlmTestService:
    def __init__(self, engine=None, history_writer=None):
        self.engine = engine
        self.history_writer = history_writer

    async def events(self, document: dict, request: LlmTestRequest):
        if self.engine is None:
            from modes import llm_service
            self.engine = llm_service
        engine = self.engine
        slot = next(slot for slot in document["draft"]["llms"] if slot["id"] == request.target)
        messages = [{"role": "user", "content": request.prompt}]
        images = [(image.b64, image.mime) for image in request.images] or None
        queue = asyncio.Queue()

        async def run():
            started = time.monotonic()
            usage = {}
            provider = next((provider for provider in document["draft"]["llmProviders"] if provider["id"] == slot["providerId"]), {})
            history = {"call_name": "LLM TEST", "task_key": "llm_test", "llm_slot": slot["id"],
                       "model": slot["model"], "service": LLM_SP["service"] if slot["id"] in LLM_SP_SLOTS else provider.get("service", ""), "input": messages,
                       "image_count": len(request.images), "output": "", "status": "cancelled"}
            with engine.settings_context(document):
                metadata_token = engine._stream_metadata_ctx.set({"task_key": "llm_test", "call_name": "LLM TEST"})
                usage_token = engine._usage_sink_ctx.set(usage)
                async def relay(event):
                    if event.get("type") == "delta":
                        history["output"] += event.get("text", "")
                        await queue.put({"type": "delta", "text": event.get("text", ""), "elapsed": time.monotonic() - started, "ttft": event.get("ttft")})
                try:
                    await queue.put({"type": "start"})
                    if slot["stream"]:
                        text = await engine.callLLMTrackedStream(messages, slot=slot["id"], images=images, stream_observer=relay,
                            metadata_sink=usage, execution_context=engine.create_llm_execution_context("llm_test", call_name="LLM TEST"))
                    else:
                        suffix = "" if slot["id"] == "llm1" else slot["id"][3:]
                        call = getattr(engine, f"callLLM{'Vision' if images else ''}{suffix}")
                        text = await call(messages, **({"images": images} if images else {}))
                    engine._fill_usage_sink_fallback(usage, text, messages)
                    history["output"] = str(text)
                    if isinstance(text, engine.ManualCancelledText):
                        await queue.put({"type": "cancelled"})
                    elif text.startswith("[LLM 실패]"):
                        history.update(status="error", error=engine._redact_in_text(text), output=engine._redact_in_text(text))
                        await queue.put({"type": "error", "error": engine._redact_in_text(text)})
                    elif not text.strip():
                        history.update(status="error", error="응답 내용이 비어 있습니다.")
                        await queue.put({"type": "error", "error": "응답 내용이 비어 있습니다. 모델명과 출력 토큰 설정을 확인하세요."})
                    else:
                        history["status"] = "ok"
                        elapsed = time.monotonic() - started
                        tokens = usage.get("completion_tokens")
                        await queue.put({"type": "done", "text": text, "elapsed": elapsed, "ttft": usage.get("ttft"),
                            "completion_tokens": tokens, "prompt_tokens": usage.get("prompt_tokens"),
                            "tps": tokens / elapsed if tokens is not None and elapsed else None})
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    error = engine._redact_in_text(str(exc))
                    history.update(status="error", error=error)
                    print(f"[LLM_TEST] {request.target}: {type(exc).__name__}: {error}")
                    await queue.put({"type": "error", "error": error or "LLM 테스트 요청을 처리하지 못했습니다."})
                finally:
                    if self.history_writer is not None:
                        elapsed = time.monotonic() - started
                        tokens = usage.get("completion_tokens")
                        history.update(elapsed=elapsed, completion_tokens=tokens,
                                       prompt_tokens=usage.get("prompt_tokens"), ttft=usage.get("ttft"),
                                       tps=tokens / elapsed if tokens is not None and elapsed else None)
                        history.update({key: usage[key] for key in (
                            "reasoning_text", "reasoning_chars", "reasoning_tokens", "tokens_estimated", "reasoning_tokens_estimated",
                        ) if key in usage})
                        try:
                            self.history_writer(history)
                        except Exception as exc:
                            print(f"[LLM_TEST] 이력 기록 실패: {type(exc).__name__}")
                    engine._usage_sink_ctx.reset(usage_token)
                    engine._stream_metadata_ctx.reset(metadata_token)
                    await queue.put(None)

        task = asyncio.create_task(run())
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    yield {"type": "ping"}
                    continue
                if event is None:
                    break
                yield event
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
