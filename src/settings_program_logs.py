"""Read the actual console streams without parsing or rewriting their output."""
from __future__ import annotations

import asyncio
import codecs
from contextlib import contextmanager
import logging
from pathlib import Path
import sys
import threading

from aiohttp import web

from app_paths import data_root


class _ConsoleCopy:
    def __init__(self, stream, output, lock):
        self.stream, self.output, self.lock = stream, output, lock

    def __getattr__(self, name):
        return getattr(self.stream, name)

    def write(self, text):
        with self.lock:
            if self.stream is not None:
                self.stream.write(text)
            if not self.output.closed:
                self.output.write(text)
                self.output.flush()
        return len(text)

    def flush(self):
        with self.lock:
            if self.stream is not None:
                self.stream.flush()
            if not self.output.closed:
                self.output.flush()


def _redirect_logging_streams(replacements) -> None:
    replacements = tuple(replacements)
    for logger in [logging.getLogger(), *logging.Logger.manager.loggerDict.values()]:
        if not isinstance(logger, logging.Logger):
            continue
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                for previous, replacement in replacements:
                    if handler.stream is previous:
                        handler.setStream(replacement)
                        break


@contextmanager
def capture_console(project_root: Path):
    """Keep console output for the current server session only."""
    if isinstance(sys.stdout, _ConsoleCopy):
        yield
        return
    path = data_root(project_root) / "logs" / "settings-console.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    output = path.open("w", encoding="utf-8", errors="backslashreplace", buffering=1)
    lock = threading.RLock()
    previous = (sys.stdout, sys.stderr)
    captured = (_ConsoleCopy(previous[0], output, lock), _ConsoleCopy(previous[1], output, lock))
    sys.stdout, sys.stderr = captured
    # Existing logging handlers may still hold the original stderr object.
    _redirect_logging_streams(zip(previous, captured))
    try:
        yield
    finally:
        sys.stdout, sys.stderr = previous
        # Include handlers created while the server was running.
        _redirect_logging_streams(zip(captured, previous))
        with lock:
            output.close()
            path.unlink(missing_ok=True)


def register_program_logs(app: web.Application, project_root: Path, schema: str, runtime, *, illustration_runtime=None) -> None:
    directory = data_root(project_root) / "logs"
    sources = [{"id": "illustration", "label": "삽화·LoRA 실행"}, {"id": "settings", "label": "설정 서버"},
               *({"id": f"comfy-{number}", "label": f"ComfyUI #{number}"} for number in (1, 2, 3))]

    def check(request):
        if request.headers.get("X-Comfy-Settings") != "1":
            raise web.HTTPForbidden()
        if request.headers.get("X-Comfy-Settings-Schema") != schema:
            raise web.HTTPConflict(text="화면을 새로고침해 주세요.")

    def file_chunk(path, offset, version, before=None):
        try:
            with path.open("rb") as stream:
                import os
                stat = os.fstat(stream.fileno())
                current_version = str(stat.st_ino)
                reset = version != current_version or offset > stat.st_size or (before is not None and before > stat.st_size)
                history = before is not None and not reset
                if reset or history:
                    end = before if history else stat.st_size
                    offset = max(0, end - 64 * 1024)
                    stream.seek(offset)
                    raw = stream.read(end - offset)
                    if offset:
                        # Prefer a complete first line; older output remains available before this cursor.
                        boundary = raw.find(b"\n")
                        if 0 <= boundary < len(raw) - 1:
                            offset += boundary + 1
                            raw = raw[boundary + 1:]
                        else:
                            while raw and raw[0] & 0xC0 == 0x80:
                                offset += 1
                                raw = raw[1:]
                else:
                    stream.seek(offset)
                    raw = stream.read(256 * 1024)
                decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
                text = decoder.decode(raw)
                consumed = len(raw) - len(decoder.getstate()[0])
                return {"text": text, "start": offset, "offset": offset + consumed, "version": current_version,
                        "reset": reset, "more": not history and consumed > 0 and offset + consumed < stat.st_size}
        except FileNotFoundError:
            return {"text": "", "start": 0, "offset": 0, "version": "", "reset": bool(version), "more": False}

    def console_chunk(source, offset, version):
        state = runtime.manager.status(instance_id=int(source[-1]), after=offset)
        current_version = str(state.get("started_at") or "")
        reset = version != current_version or state["log_reset"] or offset > state["log_seq"]
        if reset:
            state = runtime.manager.status(instance_id=int(source[-1]), after=0)
        return {"text": "".join(item["text"] for item in state["logs"]), "start": 0, "offset": state["log_seq"],
                "version": current_version, "reset": reset, "more": False}

    async def read(request):
        check(request)
        files = {"illustration": illustration_runtime.log_path if illustration_runtime is not None else directory / "internal-lora.log",
                 "settings": directory / "settings-console.log"}
        source = request.query.get("source", "illustration")
        if source not in {item["id"] for item in sources}:
            raise web.HTTPBadRequest(text="로그 종류를 확인하세요.")
        try:
            offset = max(0, int(request.query.get("offset", "0")))
            before = max(0, int(request.query["before"])) if "before" in request.query else None
        except ValueError:
            raise web.HTTPBadRequest(text="로그 위치를 확인하세요.") from None
        try:
            if request.path.endswith("/download"):
                if source in files:
                    if not files[source].is_file():
                        return web.Response(text="", headers={"Content-Disposition": f'attachment; filename="{source}.log"'})
                    return web.FileResponse(files[source], headers={"Cache-Control": "no-store",
                        "Content-Type": "text/plain; charset=utf-8", "Content-Disposition": f'attachment; filename="{source}.log"'})
                result = await asyncio.to_thread(console_chunk, source, 0, "")
                return web.Response(text=result["text"], headers={"Content-Disposition": f'attachment; filename="{source}.log"'})
            version = request.query.get("version", "")
            result = await asyncio.to_thread(file_chunk, files[source], offset, version, before) if source in files else await asyncio.to_thread(console_chunk, source, offset, version)
            return web.json_response({**result, "sources": sources}, headers={"Cache-Control": "no-store"})
        except OSError:
            return web.json_response({"error": "로그 파일을 읽지 못했습니다."}, status=503)

    async def browser_output(request):
        check(request)
        if request.content_type != "application/json" or (request.content_length or 0) > 64 * 1024:
            raise web.HTTPBadRequest()
        body = await request.json()
        if not isinstance(body, dict) or not isinstance(body.get("text"), str) or len(body["text"]) > 24000:
            raise web.HTTPBadRequest()
        page = str(body.get("page", ""))[:250].replace("\n", " ").replace("\r", " ")
        print(f"[BROWSER {page}]\n{body['text']}", file=sys.stderr, flush=True)
        return web.json_response({"ok": True}, headers={"Cache-Control": "no-store"})

    app.router.add_get("/api/setting/program-logs", read)
    app.router.add_get("/api/setting/program-logs/download", read)
    app.router.add_post("/api/setting/program-logs/browser", browser_output)
