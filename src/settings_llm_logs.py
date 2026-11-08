"""Settings views over the original LLM history and active stream controls."""
from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
import sys
import uuid

import aiohttp
from aiohttp import web

from app_paths import data_root
from llm_token_usage import output_token_stats


SUMMARY_FIELDS = (
    "ts", "call_name", "task_key", "model", "service", "status", "llm_slot",
    "prompt_tokens", "completion_tokens", "elapsed", "tps", "ttft",
    "reasoning_tokens", "reasoning_chars", "tokens_estimated", "reasoning_tokens_estimated",
    "race_id", "race_role", "race_status", "history_id",
)


def register_llm_logs(app: web.Application, project_root: Path, schema: str, runtime):
    history_path = data_root(project_root) / "logs" / "lighbd_history.jsonl"
    settings_history_path = history_path.with_name("settings_llm_history.jsonl")
    history_lock = asyncio.Lock()
    history_stamp = None
    records_by_id: dict[str, dict] = {}
    summaries: list[dict] = []

    def read_history():
        from modes.lighbd_service import _load_lighbd_history, LIGHBD_HISTORY_MAX
        records = _load_lighbd_history(history_path=history_path, strict=True)
        records += _load_lighbd_history(history_path=settings_history_path, strict=True)
        records.sort(key=lambda record: str(record.get("ts") or ""))
        records = records[-LIGHBD_HISTORY_MAX:]
        indexed = {}
        rows = []
        occurrences = {}
        for record in reversed(records):
            serialized = json.dumps(record, ensure_ascii=False, sort_keys=True)
            identity = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
            # Identical records still have distinct rows, stable across refreshes.
            occurrences[identity] = occurrences.get(identity, 0) + 1
            identity += f"-{occurrences[identity]}"
            # Older logs can retain thinking text without a separate token count.
            if record.get("reasoning_tokens") is None:
                stats = output_token_stats(
                    "", str(record.get("reasoning_text") or record.get("reasoning") or ""),
                    completion_tokens=record.get("completion_tokens"),
                    reasoning_chars=record.get("reasoning_chars") or 0,
                )
                record = {**record, **{key: value for key, value in stats.items()
                                      if key.startswith("reasoning_")}}
            indexed[identity] = {**record, "id": identity}
            row = {key: record[key] for key in SUMMARY_FIELDS if key in record}
            row.update(id=identity, preview=str(record.get("error") or record.get("output") or "")[:240])
            rows.append(row)
        return indexed, rows

    def record_history(record):
        from modes.lighbd_service import _log_lighbd_history
        # Each process owns its history file; the view combines the two sources.
        _log_lighbd_history({"history_id": uuid.uuid4().hex, **record}, history_path=settings_history_path)

    def guard(request):
        if request.headers.get("X-Comfy-Settings") != "1":
            raise web.HTTPForbidden()
        if request.headers.get("X-Comfy-Settings-Schema") != schema:
            raise web.HTTPConflict(text=json.dumps({"error": "설정 화면을 새로고침해 주세요."}, ensure_ascii=False), content_type="application/json")

    def response(value, status=200):
        return web.json_response(value, status=status, headers={"Cache-Control": "no-store"})

    async def history(request):
        nonlocal history_stamp, records_by_id, summaries
        guard(request)
        try:
            async with history_lock:
                stamp = []
                for path in (history_path, settings_history_path):
                    try:
                        stat = path.stat()
                        stamp.append((stat.st_mtime_ns, stat.st_size))
                    except FileNotFoundError:
                        stamp.append((0, 0))
                if history_stamp != stamp:
                    records_by_id, summaries = await asyncio.to_thread(read_history)
                    history_stamp = stamp
                identity = request.match_info.get("identity")
                if identity:
                    record = records_by_id.get(identity)
                    if record is None:
                        return response({"error": "보관 기간이 지나거나 갱신된 로그입니다. 목록에서 다시 선택하세요."}, 404)
                    return response({"record": record})
                return response({"history": summaries, "count": len(summaries)})
        except (OSError, ValueError, ImportError):
            return response({"error": "LLM 로그를 불러올 수 없습니다. 잠시 후 다시 시도하세요."}, 503)

    async def streams(request):
        guard(request)
        engine = sys.modules.get("modes.llm_service")
        local = engine.get_active_streams() if engine else []
        result = [{**stream, "source": "settings"} for stream in local]
        errors = []
        try:
            worker = await runtime.llm_streams()
            result.extend({**stream, "source": "internal-lora"} for stream in worker["streams"])
        except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError, ValueError):
            errors.append("작업 라이브에 연결할 수 없습니다. 자동으로 다시 연결합니다.")
        return response({"streams": result, "errors": errors})

    async def control(request):
        guard(request)
        if request.content_type != "application/json":
            raise web.HTTPUnsupportedMediaType()
        try:
            body = await request.json()
            action = body.get("action") if isinstance(body, dict) else None
            if action not in {"cancel", "retry", "parallel_retry", "use_partial"}:
                return response({"error": "지원하지 않는 라이브 동작입니다."}, 400)
            stream_id = request.match_info["stream_id"]
            source = request.match_info["source"]
            if source == "settings":
                engine = sys.modules.get("modes.llm_service")
                if engine is None:
                    return response({"error": "활성 스트림을 찾을 수 없습니다."}, 404)
                success, message = engine.request_stream_control(stream_id, action)
                return response({"success": True} if success else {"error": message}, 200 if success else 400)
            return response(await runtime.llm_streams(stream_id, action))
        except (ValueError, TypeError) as exc:
            return response({"error": str(exc) or "라이브 요청을 확인하세요."}, 400)
        except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError):
            return response({"error": "라이브에 연결할 수 없습니다. 다시 시도하세요."}, 503)

    app.router.add_get("/api/setting/llm/history", history)
    app.router.add_get("/api/setting/llm/history/{identity}", history)
    app.router.add_get("/api/setting/llm/streams", streams)
    app.router.add_post("/api/setting/llm/streams/{source:settings|internal-lora}/{stream_id}/control", control)
    return record_history
