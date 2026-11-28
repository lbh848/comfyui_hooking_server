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

from app_paths import config_root, data_root
from llm_settings import RuntimeSettingsReader, SLOT_IDS
from llm_token_usage import output_token_stats
from llm_failure_details import format_llm_failure


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
    file_cache: dict[Path, dict] = {}
    settings_reader = RuntimeSettingsReader(config_root(project_root))

    def read_llm_config() -> dict:
        try:
            return settings_reader.read()
        except (OSError, ValueError, RuntimeError):
            return {}

    def model_alias(record: dict, config: dict) -> str:
        """Resolve the user-facing alias for the slot that produced a record."""
        slot = str(record.get("llm_slot") or "").strip().lower()
        model = str(record.get("model") or "").strip()
        if slot not in SLOT_IDS:
            # Old records may lack a slot; mirror the original UI and only use an
            # alias when exactly one configured slot matches the recorded model.
            matches = [
                candidate for candidate in SLOT_IDS
                if model and str(config.get(
                    f"llm_model{'' if candidate == 'llm1' else candidate[3:]}", "")).strip() == model
            ]
            slot = matches[0] if len(matches) == 1 else ""
        if not slot:
            return ""
        suffix = "" if slot == "llm1" else slot[3:]
        return str(config.get(f"llm_alias{suffix}", "") or "").strip()

    def build_entry(raw_record: dict) -> tuple[dict, str, dict]:
        serialized = json.dumps(raw_record, ensure_ascii=False, sort_keys=True)
        base_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        record = raw_record
        # Older logs can retain thinking text without a separate token count.
        if record.get("reasoning_tokens") is None:
            stats = output_token_stats(
                "", str(record.get("reasoning_text") or record.get("reasoning") or ""),
                completion_tokens=record.get("completion_tokens"),
                reasoning_chars=record.get("reasoning_chars") or 0,
            )
            record = {**record, **{key: value for key, value in stats.items()
                                  if key.startswith("reasoning_")}}
        # Existing history already contains finish details; show them without
        # rewriting logs or changing identities used by history links.
        if record.get("status") == "error":
            record = {**record, "error": format_llm_failure(record.get("error", ""), record)}
        base_row = {key: record[key] for key in SUMMARY_FIELDS if key in record}
        base_row["preview"] = str(record.get("error") or record.get("output") or "")[:240]
        return record, base_hash, base_row

    def load_file_entries(path: Path, loader, *, use_incremental: bool) -> list[tuple[dict, str, dict]]:
        if not use_incremental:
            return [build_entry(record) for record in loader(history_path=path, strict=True)]
        try:
            stat = path.stat()
        except FileNotFoundError:
            file_cache.pop(path, None)
            return []
        size = stat.st_size
        mtime_ns = stat.st_mtime_ns
        cached = file_cache.get(path)
        if cached and cached["mtime_ns"] == mtime_ns and cached["size"] == size:
            return cached["entries"]

        if (
            cached
            and cached["size"] > 0
            and size > cached["size"]
            and cached["tail"].endswith(b"\n")
        ):
            try:
                with path.open("rb") as handle:
                    head = handle.read(len(cached["head"]))
                    handle.seek(cached["size"] - len(cached["tail"]))
                    tail = handle.read(len(cached["tail"]))
                    if head == cached["head"] and tail == cached["tail"]:
                        appended_bytes = handle.read(size - cached["size"])
                        new_entries = list(cached["entries"])
                        for raw_line in appended_bytes.decode("utf-8").splitlines():
                            line = raw_line.strip()
                            if not line:
                                continue
                            try:
                                parsed = json.loads(line)
                            except Exception:
                                continue
                            new_entries.append(build_entry(parsed))
                        new_tail_window = (tail + appended_bytes)[-256:]
                        file_cache[path] = {
                            "mtime_ns": mtime_ns,
                            "size": size,
                            "head": head,
                            "tail": new_tail_window,
                            "entries": new_entries,
                        }
                        return new_entries
            except OSError:
                raise
            except UnicodeDecodeError:
                raise

        with path.open("rb") as handle:
            raw_bytes = handle.read()
        entries: list[tuple[dict, str, dict]] = []
        for raw_line in raw_bytes.decode("utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except Exception:
                continue
            entries.append(build_entry(parsed))
        file_cache[path] = {
            "mtime_ns": mtime_ns,
            "size": len(raw_bytes),
            "head": raw_bytes[:128],
            "tail": raw_bytes[-256:],
            "entries": entries,
        }
        return entries

    def read_history():
        from modes.lighbd_service import _load_lighbd_history, LIGHBD_HISTORY_MAX
        use_incremental = (
            getattr(_load_lighbd_history, "__module__", "") == "modes.lighbd_service"
            and getattr(_load_lighbd_history, "__name__", "") == "_load_lighbd_history"
        )
        entries = list(load_file_entries(history_path, _load_lighbd_history, use_incremental=use_incremental))
        entries.extend(load_file_entries(settings_history_path, _load_lighbd_history, use_incremental=use_incremental))
        entries.sort(key=lambda entry: str(entry[0].get("ts") or ""))
        entries = entries[-LIGHBD_HISTORY_MAX:]
        config = read_llm_config()
        indexed = {}
        rows = []
        occurrences = {}
        for record, base_hash, base_row in reversed(entries):
            # Identical records still have distinct rows, stable across refreshes.
            occurrences[base_hash] = occurrences.get(base_hash, 0) + 1
            identity = f"{base_hash}-{occurrences[base_hash]}"
            alias = model_alias(record, config)
            indexed[identity] = {**record, "id": identity, "llm_alias": alias}
            rows.append({**base_row, "id": identity, "llm_alias": alias})
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
        monitor = request.query.get("monitor") == "1"
        engine = sys.modules.get("modes.llm_service")
        local = engine.get_active_streams(monitor=monitor) if engine else []
        config = {} if monitor else read_llm_config()
        result = [{**stream, "source": "settings", "llm_alias": model_alias(stream, config)}
                  for stream in local]
        errors = []
        try:
            worker = await runtime.llm_streams(monitor=monitor)
            result.extend({**stream, "source": "internal-lora", "llm_alias": model_alias(stream, config)}
                          for stream in worker["streams"])
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
