"""Run the fixed Gemma image-analysis workflow through the local Comfy queue."""
from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import json
import logging
import os
import re
import secrets
from contextlib import suppress
from pathlib import Path
import time
import uuid

import aiohttp

from app_paths import SOURCE_DIR, config_root, data_root
from llm_settings import LLM_SP, LLM_SP_SLOTS, read_object, validate_context_size


def build_request(messages: list) -> dict:
    """Convert the engine's multimodal message format to the node's JSON contract."""
    paragraphs, images, history_ranges = [], [], []
    prompt_length = 0
    for message in messages:
        content = message.get("content", "")
        parts = [{"type": "text", "text": content}] if isinstance(content, str) else content
        text, ranges = [], []
        text_length = 0
        for part in parts or []:
            if part.get("type") == "text":
                value = part.get("text", "")
                for start, end, label in getattr(value, "variable_spans", ()):
                    if label in {"과거 대화", "대사 귀속용 과거 대화"}:
                        ranges.append((text_length + start, text_length + end))
                text.append(str(value))
                text_length += len(str(value)) + 1
            elif part.get("type") == "image_url":
                image_url = part.get("image_url", {})
                url = image_url.get("url", "") if isinstance(image_url, dict) else image_url
                if not isinstance(url, str) or not url.startswith("data:image/") or ";base64," not in url:
                    raise ValueError("LLM_SP에는 Base64 이미지 데이터를 전달하세요.")
                header, encoded = url.split(",", 1)
                try:
                    raw = base64.b64decode(encoded, validate=True)
                except (ValueError, binascii.Error):
                    raise ValueError("LLM_SP 이미지의 Base64 형식이 올바르지 않습니다.") from None
                if not raw:
                    raise ValueError("LLM_SP 이미지가 비어 있습니다.")
                images.append({"encoding": "base64", "media_type": header[5:].split(";")[0],
                               "sha256": hashlib.sha256(raw).hexdigest(), "data": encoded})
            else:
                raise ValueError("LLM_SP는 텍스트와 이미지 입력만 지원합니다.")
        if text:
            joined = "\n".join(text)
            paragraph = joined.strip()
            if paragraph:
                prefix = f"{message.get('role', 'user').upper()}:\n" if len(messages) > 1 else ""
                left = len(joined) - len(joined.lstrip())
                right = left + len(paragraph)
                for start, end in ranges:
                    if max(start, left) < min(end, right):
                        history_ranges.append([prompt_length + len(prefix) + max(start, left) - left,
                                               prompt_length + len(prefix) + min(end, right) - left])
                paragraphs.append(prefix + paragraph)
                prompt_length += len(prefix) + len(paragraph) + 2
    if len(images) > 1:
        raise ValueError("LLM_SP에는 이미지를 최대 1장까지 추가할 수 있습니다.")
    if not paragraphs:
        raise ValueError("LLM_SP 분석 프롬프트를 입력하세요.")
    payload = {"schema": "soya.gemma4.vision.v1", "prompt": "\n\n".join(paragraphs)}
    if images:
        payload["image"] = images[0]
    if history_ranges:
        payload["history_ranges"] = history_ranges
    return payload


async def analyze(messages: list, *, slot_id: str = LLM_SP["id"], reasoning_budget: int = 2048,
                  context_size: int = LLM_SP["context_size"],
                  vram_headroom_mib: int = LLM_SP["vram_headroom_mib"],
                  project_root: Path = SOURCE_DIR, on_event=None, metadata=None, preload_only: bool = False) -> str:
    started = time.monotonic()
    metadata = metadata if metadata is not None else {}
    profile = LLM_SP_SLOTS[slot_id]
    context_size = validate_context_size(context_size)
    if type(reasoning_budget) is not int or reasoning_budget not in LLM_SP["reasoning_budgets"]:
        raise ValueError("Gemma reasoning budget must be 1024, 2048, or 4096 tokens")
    payload = {} if preload_only else build_request(messages)
    from configuration import config_path, read_config
    config = await asyncio.to_thread(read_config, config_path(project_root))
    configured_path = config.get(LLM_SP["workflow_binding"])
    workflow_path = Path(configured_path) if configured_path else data_root(project_root) / "comfy/user/default/workflows" / LLM_SP["workflow_filename"]
    if not workflow_path.is_file():
        raise ValueError(f"{LLM_SP['workflow_name']} 워크플로우를 먼저 설치하세요.")
    workflow = await asyncio.to_thread(read_object, workflow_path)
    # Conversion only needs the graph structure. Its endpoint has a 1 MiB limit,
    # so neither the example image nor this request belongs in that upload.
    targets = [node for node in workflow.get("nodes", []) if node.get("type") == LLM_SP["node_type"]]
    if len(targets) != 1:
        raise ValueError("고정 워크플로우에 Gemma 이미지 분석 노드가 하나 있어야 합니다.")
    targets[0]["widgets_values"][0] = ""
    targets[0]["widgets_values"][1] = context_size
    node_id = str(targets[0]["id"])
    port = int(config.get("comfyui_port", 8188))
    host = os.environ.get("REAL_COMFY_HOST", "127.0.0.1")
    base_url = f"http://{host}:{port}"
    prompt_id = None
    client_id = str(uuid.uuid4())
    websocket = None
    live_identity = None
    live_started_at_ms = None
    live_terminal = False
    retired_executions = set()
    last_sequence = -1
    emitted = b""
    emitted_reasoning = b""
    completed = False
    timeout = aiohttp.ClientTimeout(total=30, connect=5)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async def request(method, path, **kwargs):
            async with session.request(method, base_url + path, **kwargs) as response:
                if response.status != 200:
                    # Node validation responses can echo the image and prompt.
                    if response.status == 413:
                        raise RuntimeError(f"ComfyUI {path} 요청 크기가 서버의 허용 한도를 초과했습니다. (HTTP 413)")
                    raise RuntimeError(f"ComfyUI {path} 요청 실패 (HTTP {response.status}). 워크플로우와 노드 설치를 확인하세요.")
                return await response.json()

        async def emit_text(text):
            if text and on_event is not None:
                metadata.setdefault("ttft", time.monotonic() - started)
                await on_event({"type": "delta", "text": text, "ttft": metadata["ttft"]})

        async def emit_reasoning(text):
            if text and on_event is not None:
                metadata.setdefault("ttft", time.monotonic() - started)
                await on_event({"type": "delta", "reasoning": text, "ttft": metadata["ttft"]})

        def capture_usage(usage):
            if not isinstance(usage, dict):
                return
            # llama.cpp completion_tokens already includes thinking tokens.
            for key in ("completion_tokens", "prompt_tokens", "total_tokens"):
                value = usage.get(key)
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    metadata[key] = value

        async def receive_snapshot(snapshot):
            nonlocal live_identity, last_sequence, emitted, emitted_reasoning
            nonlocal live_started_at_ms, live_terminal
            if (not isinstance(snapshot, dict) or snapshot.get("prompt_id") != prompt_id
                    or str(snapshot.get("node_id")) != node_id):
                return
            execution_id = snapshot.get("execution_id")
            sequence = snapshot.get("sequence")
            if not isinstance(execution_id, str) or not isinstance(sequence, int):
                return
            if execution_id in retired_executions:
                return
            started_at_ms = snapshot.get("started_at_ms")
            if live_identity is not None and execution_id != live_identity["execution_id"]:
                # The Gemma node counts input (possibly multiple times while
                # fitting history) before generating. Each count publishes its
                # own execution ID for the same Comfy prompt and node. Follow
                # the next execution until actual output has begun.
                newer = (type(started_at_ms) is int and type(live_started_at_ms) is int
                         and started_at_ms > live_started_at_ms)
                if emitted or emitted_reasoning or not (live_terminal or newer):
                    return
                retired_executions.add(live_identity["execution_id"])
                last_sequence = -1
                live_started_at_ms = None
            if sequence <= last_sequence:
                return
            live_identity = {"execution_id": execution_id, "prompt_id": prompt_id, "node_id": node_id}
            if live_started_at_ms is None:
                live_started_at_ms = started_at_ms
            live_terminal = snapshot.get("terminal") is True
            last_sequence = sequence
            capture_usage(snapshot.get("usage"))
            if snapshot.get("finish_reason"):
                metadata["finish_reason"] = snapshot["finish_reason"]
            for pane_name in ("thinking", "response"):
                pane = snapshot.get(pane_name) or {}
                tail = str(pane.get("text") or "").encode("utf-8")
                total = pane.get("total_bytes", len(tail))
                if not isinstance(total, int):
                    continue
                previous = emitted_reasoning if pane_name == "thinking" else emitted
                offset = total - len(tail)
                # Cumulative UTF-8 tails may overlap. History fills reconnect gaps.
                if 0 <= offset <= len(previous) < total and tail.startswith(previous[offset:]):
                    delta = tail[len(previous) - offset:]
                    if pane_name == "thinking":
                        emitted_reasoning += delta
                        metadata["reasoning_text"] = emitted_reasoning.decode("utf-8")
                        metadata["reasoning_chars"] = len(metadata["reasoning_text"])
                        await emit_reasoning(delta.decode("utf-8"))
                    else:
                        emitted += delta
                        await emit_text(delta.decode("utf-8"))

        try:
            async with asyncio.timeout(1800):
                graph = await request("POST", "/workflow/convert", json=workflow)
                if not isinstance(graph, dict) or graph.get(node_id, {}).get("class_type") != LLM_SP["node_type"]:
                    raise RuntimeError("Gemma 워크플로우를 API 형식으로 변환하지 못했습니다.")
                # Attach the JSON/Base64 request only to the execution endpoint.
                # The slot owns context, reasoning, and VRAM headroom. Other node options come from the workflow.
                graph[node_id]["inputs"]["prompt_json"] = json.dumps(payload, ensure_ascii=False)
                graph[node_id]["inputs"]["context_size"] = context_size
                graph[node_id]["inputs"]["vram_headroom_mib"] = vram_headroom_mib
                graph[node_id]["inputs"]["thinking_mode"] = profile["thinking_mode"]
                graph[node_id]["inputs"]["reasoning_budget"] = str(reasoning_budget)
                graph[node_id]["inputs"]["preload_only"] = preload_only
                # API submissions do not run the UI's seed randomization widget.
                seed = secrets.randbelow(2 ** 31)
                graph[node_id]["inputs"]["seed"] = seed
                metadata["seed"] = seed
                logging.getLogger(__name__).info("Gemma request: slot=%s seed=%s", slot_id, seed)
                if on_event is not None:
                    websocket = await session.ws_connect(base_url + "/ws", params={"clientId": client_id}, heartbeat=20)
                # Know the exact job ID even if the submit response is interrupted.
                prompt_id = str(uuid.uuid4())
                submitted = await request("POST", "/prompt", json={"prompt": graph, "client_id": client_id, "prompt_id": prompt_id})
                if submitted.get("prompt_id") != prompt_id:
                    raise RuntimeError("ComfyUI가 Gemma 작업 ID를 올바르게 반환하지 않았습니다.")
                next_history = time.monotonic() + 0.5
                next_connect = 0.0
                while True:
                    now = time.monotonic()
                    if on_event is not None and websocket is None and now >= next_connect:
                        next_connect = now + 2
                        try:
                            websocket = await session.ws_connect(base_url + "/ws", params={"clientId": client_id}, heartbeat=20)
                            restored = await request("GET", "/llamacpp/generation/active", params={"client_id": client_id})
                            for snapshot in restored.get("executions", []):
                                await receive_snapshot(snapshot)
                        except (aiohttp.ClientError, RuntimeError, TimeoutError):
                            if websocket is not None:
                                await websocket.close()
                                websocket = None
                    if websocket is not None:
                        try:
                            event = await websocket.receive(timeout=0.5)
                            if event.type == aiohttp.WSMsgType.TEXT:
                                frame = event.json()
                                if frame.get("type") == "llamacpp.generation":
                                    await receive_snapshot(frame.get("data"))
                            elif event.type in {aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR}:
                                await websocket.close()
                                websocket = None
                        except TimeoutError:
                            pass
                        except aiohttp.ClientError:
                            await websocket.close()
                            websocket = None
                    else:
                        await asyncio.sleep(0.5)
                    if time.monotonic() < next_history:
                        continue
                    next_history = time.monotonic() + 1
                    history = await request("GET", f"/history/{prompt_id}")
                    record = history.get(prompt_id)
                    if record:
                        status = record.get("status", {})
                        if status.get("status_str") == "error":
                            completed = True
                            details = next((item[1].get("exception_message", "") for item in status.get("messages", [])
                                            if item[0] == "execution_error"), "")
                            overflow = re.search(
                                r"request input plus requested output exceeds the effective slot context by (\d+) tokens",
                                str(details),
                            )
                            if overflow:
                                raise RuntimeError(
                                    f"Gemma 컨텍스트 한도 초과: 입력과 요청한 응답이 한도를 {int(overflow[1]):,}토큰 초과했습니다. "
                                    f"{profile['label']}의 '컨텍스트'를 늘리거나 입력을 줄이세요."
                                )
                            raise RuntimeError(f"Gemma 워크플로우 실행 실패: {str(details)[:800] or 'Comfy 실행 로그를 확인하세요.'}")
                        node_output = record.get("outputs", {}).get(node_id, {})
                        output = node_output.get("text", [])
                        if status.get("completed"):
                            completed = True
                            details = node_output.get("soya_gemma4", [])
                            if details and isinstance(details[0], dict):
                                details = details[0]
                                capture_usage(details.get("usage"))
                                if isinstance(details.get("context_budget"), dict):
                                    metadata["context_budget"] = details["context_budget"]
                                if details.get("finish_reason"):
                                    metadata["finish_reason"] = details["finish_reason"]
                                reasoning = str(details.get("reasoning_text") or "")
                                metadata.update(reasoning_text=reasoning, reasoning_chars=len(reasoning))
                                full_reasoning = reasoning.encode("utf-8")
                                if full_reasoning.startswith(emitted_reasoning):
                                    await emit_reasoning(full_reasoning[len(emitted_reasoning):].decode("utf-8"))
                            text = output if isinstance(output, str) else "\n".join(output)
                            if not text.strip():
                                raise RuntimeError("Gemma 워크플로우의 분석 응답이 비어 있습니다.")
                            full_text = text.encode("utf-8")
                            if full_text.startswith(emitted):
                                await emit_text(full_text[len(emitted):].decode("utf-8"))
                            metadata["elapsed"] = time.monotonic() - started
                            if metadata.get("completion_tokens") is not None:
                                metadata["tps"] = metadata["completion_tokens"] / max(metadata["elapsed"], 0.001)
                            return text
        except aiohttp.ClientConnectorError:
            raise RuntimeError(f"Comfy #1에 연결할 수 없습니다. Comfy 실행 관리에서 Comfy #1을 실행하세요. (포트 {port})") from None
        except TimeoutError:
            raise RuntimeError("Gemma 워크플로우 응답 시간이 초과되었습니다. Comfy 실행 로그를 확인하세요.") from None
        finally:
            if prompt_id and not completed:
                if live_identity and websocket is not None and not websocket.closed:
                    try:
                        async with session.post(base_url + "/llamacpp/generation/cancel",
                                                params={"client_id": client_id}, json=live_identity,
                                                timeout=aiohttp.ClientTimeout(total=5)) as response:
                            if response.status not in (200, 202, 404, 409):
                                response.raise_for_status()
                    except (aiohttp.ClientError, TimeoutError):
                        logging.warning("LLM_SP generation cancellation could not be delivered: %s", prompt_id)
                try:
                    # Cancel only our job, including a job still waiting in the queue.
                    async with session.post(base_url + f"/api/jobs/{prompt_id}/cancel",
                                            timeout=aiohttp.ClientTimeout(total=5)) as response:
                        response.raise_for_status()
                except (aiohttp.ClientError, TimeoutError):
                    logging.warning("LLM_SP job cancellation could not be delivered: %s", prompt_id)
            if websocket is not None:
                await websocket.close()


async def stream(messages: list, *, slot_id: str = LLM_SP["id"], reasoning_budget: int = 2048,
                 context_size: int = LLM_SP["context_size"],
                 vram_headroom_mib: int = LLM_SP["vram_headroom_mib"],
                 project_root: Path = SOURCE_DIR):
    """Keep HTTP deadlines and cancellation in one task while callers consume deltas."""
    events = asyncio.Queue()

    async def run():
        metadata = {}
        try:
            text = await analyze(messages, slot_id=slot_id, reasoning_budget=reasoning_budget,
                                 context_size=context_size, vram_headroom_mib=vram_headroom_mib,
                                 project_root=project_root, on_event=events.put, metadata=metadata)
            await events.put({"type": "done", "text": text, **metadata})
        except Exception as exc:
            await events.put({"type": "error", "error": str(exc)})
        finally:
            events.put_nowait(None)

    task = asyncio.create_task(run())
    try:
        yield {"type": "start"}
        while (event := await events.get()) is not None:
            yield event
    finally:
        if not task.done():
            task.cancel()
        with suppress(asyncio.CancelledError):
            await task
