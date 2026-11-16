"""A dedicated, local Chromium page rendered inside the workshop.

Only this application's browser profile is controlled; personal browser windows
and the desktop are never captured. CDP stays on loopback and is not proxied.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import os
from pathlib import Path
import shutil
import subprocess

import aiohttp
from aiohttp import web

from app_paths import data_root

HOME = "https://shima.donmai.us/tags"
PAGE_CONTROLS = """(() => {
    if (!location.hostname.endsWith('.donmai.us') || !CSS.supports('appearance', 'base-select')) return;
    const install = () => {
        if (document.getElementById('comfy-danbooru-controls')) return;
        // Preserve the site's collapsed field width when the picker changes renderer.
        for (const select of document.querySelectorAll('select:not([multiple])')) {
            if (select.size <= 1) select.style.minInlineSize = getComputedStyle(select).width;
        }
        const style = document.createElement('style');
        style.id = 'comfy-danbooru-controls';
        style.textContent = `
            select:not([multiple]):not([size]), select:not([multiple])[size="1"],
            select:not([multiple]):not([size])::picker(select), select:not([multiple])[size="1"]::picker(select) {
                appearance: base-select !important;
            }
            select::picker(select) {
                max-block-size: min(320px, calc(100dvh - 16px));
                overflow-y: auto;
            }
        `;
        (document.head || document.documentElement).append(style);
    };
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', install, {once: true});
    else install();
})()"""
CURSORS = set("default none context-menu help pointer progress wait cell crosshair text vertical-text alias copy move no-drop not-allowed grab grabbing all-scroll col-resize row-resize n-resize e-resize s-resize w-resize ne-resize nw-resize se-resize sw-resize ew-resize ns-resize nesw-resize nwse-resize zoom-in zoom-out".split())
CURSOR_AT = """([x, y]) => {
    let element = document.elementFromPoint(x, y);
    while (element?.shadowRoot) {
        const child = element.shadowRoot.elementFromPoint(x, y);
        if (!child || child === element) break;
        element = child;
    }
    if (!element) return 'default';
    const cursor = getComputedStyle(element).cursor.split(',').at(-1).trim();
    if (cursor && cursor !== 'auto') return cursor;
    if (element.closest(':disabled, [aria-disabled="true"]')) return 'not-allowed';
    if (element.isContentEditable || element.matches('textarea, input:not([type]), input[type="text"], input[type="search"], input[type="email"], input[type="url"], input[type="password"], input[type="tel"], input[type="number"]')) return 'text';
    if (element.closest('a[href], button, select, summary, label[for], input, [role="button"], [role="link"]')) return 'pointer';
    const caret = document.caretRangeFromPoint?.(x, y);
    if (caret?.startContainer.nodeType === Node.TEXT_NODE && element.contains(caret.startContainer)) {
        const range = document.createRange(), node = caret.startContainer, offset = caret.startOffset;
        range.setStart(node, Math.max(0, offset - 1));
        range.setEnd(node, Math.min(node.length, offset + 1));
        if (Array.from(range.getClientRects()).some(r => x >= r.left && x <= r.right && y >= r.top && y <= r.bottom)) return 'text';
    }
    return 'default';
}"""


def browser_executable() -> str:
    for folder, suffix in (
        (os.environ.get("PROGRAMFILES", ""), "Google/Chrome/Application/chrome.exe"),
        (os.environ.get("PROGRAMFILES(X86)", ""), "Microsoft/Edge/Application/msedge.exe"),
        (os.environ.get("LOCALAPPDATA", ""), "Google/Chrome/Application/chrome.exe"),
    ):
        candidate = Path(folder) / suffix
        if folder and candidate.is_file():
            return str(candidate)
    for name in ("google-chrome", "chromium", "chromium-browser", "microsoft-edge"):
        if found := shutil.which(name):
            return found
    raise RuntimeError("Chrome 또는 Microsoft Edge를 설치한 뒤 다시 열어 주세요.")


class DanbooruBrowser:
    def __init__(self, root):
        self.profile = data_root(root) / "danbooru_browser"
        self.process = None
        self.client = None
        self.socket = None
        self.reader = None
        self.pending = {}
        self.sequence = 0
        self.lock = asyncio.Lock()
        self.interaction_lock = asyncio.Lock()
        self.width, self.height = 1100, 760
        self.port = 0
        self.target = ""
        self.viewers: dict[web.WebSocketResponse, asyncio.Queue] = {}
        self.streaming = False
        self.last_frame = None
        self.pointer_position = None
        self.pointer_revision = 0

    @property
    def running(self):
        return self.socket is not None and not self.socket.closed

    async def _read(self):
        try:
            async for message in self.socket:
                if message.type != aiohttp.WSMsgType.TEXT:
                    continue
                payload = json.loads(message.data)
                if payload.get("method") == "Page.screencastFrame":
                    params = payload["params"]
                    self.sequence += 1
                    # Acknowledge without waiting on this reader's own response loop.
                    await self.socket.send_json({"id": self.sequence, "method": "Page.screencastFrameAck", "params": {"sessionId": params["sessionId"]}})
                    metadata = params["metadata"]
                    self.last_frame = {"frame": params["data"], "width": metadata["deviceWidth"], "height": metadata["deviceHeight"]}
                    for queue in self.viewers.values():
                        if queue.full():
                            queue.get_nowait()
                        queue.put_nowait(self.last_frame)
                future = self.pending.pop(payload.get("id"), None)
                if future is not None and not future.done():
                    if "error" in payload:
                        logging.getLogger(__name__).warning("Danbooru browser command failed: %s", payload["error"])
                        future.set_exception(RuntimeError("브라우저 작업을 완료하지 못했습니다. 다시 시도해 주세요."))
                    else:
                        future.set_result(payload.get("result", {}))
        finally:
            for future in self.pending.values():
                if not future.done():
                    future.set_exception(RuntimeError("단부르 브라우저 연결이 끊겼습니다. 다시 열어 주세요."))
            self.pending.clear()

    async def command(self, method, params=None, timeout=15):
        if not self.running:
            raise RuntimeError("단부르 브라우저를 먼저 열어 주세요.")
        self.sequence += 1
        identifier = self.sequence
        future = asyncio.get_running_loop().create_future()
        self.pending[identifier] = future
        try:
            await self.socket.send_json({"id": identifier, "method": method, "params": params or {}})
            return await asyncio.wait_for(future, timeout)
        except asyncio.TimeoutError:
            raise RuntimeError("단부르 브라우저의 응답이 늦습니다. 잠시 후 다시 시도해 주세요.") from None
        finally:
            self.pending.pop(identifier, None)

    async def evaluate(self, expression):
        result = await self.command("Runtime.evaluate", {
            "expression": expression, "returnByValue": True, "awaitPromise": True,
        })
        if result.get("exceptionDetails"):
            raise RuntimeError("단부르 페이지를 읽지 못했습니다. 페이지가 열린 뒤 다시 시도해 주세요.")
        return result.get("result", {}).get("value")

    async def navigate(self, url):
        result = await self.command("Page.navigate", {"url": url})
        if result.get("errorText"):
            raise RuntimeError("단부르 웹페이지에 접속하지 못했습니다. 네트워크 상태를 확인하고 새로고침해 주세요.")

    async def start_stream(self):
        if self.running and self.viewers and not self.streaming:
            # Evaluation waits for the new document's execution context after navigation.
            await self.evaluate("document.readyState")
            await self.command("Page.startScreencast", {"format": "jpeg", "quality": 80, "everyNthFrame": 1, "maxWidth": 1440, "maxHeight": 1000})
            self.streaming = True

    async def read_cursor(self):
        if not self.pointer_position:
            return {"cursor": "default", "cursor_revision": self.pointer_revision}
        revision = self.pointer_revision
        cursor = await self.evaluate(f"({CURSOR_AT})({json.dumps(self.pointer_position)})")
        return {"cursor": cursor if cursor in CURSORS else "default", "cursor_revision": revision}

    async def start(self):
        async with self.lock:
            if self.running:
                return
            await self.close()
            executable = browser_executable()
            self.profile.mkdir(parents=True, exist_ok=True)
            active_port = self.profile / "DevToolsActivePort"
            active_port.unlink(missing_ok=True)
            startupinfo = None
            if os.name == "nt":
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = 0
            self.process = subprocess.Popen([
                executable, f"--user-data-dir={self.profile}", "--remote-debugging-port=0",
                "--remote-debugging-address=127.0.0.1", "--no-first-run", "--no-default-browser-check",
                "--disable-background-timer-throttling", "--disable-renderer-backgrounding",
                "--window-size=1120,850", "about:blank",
            ], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                startupinfo=startupinfo)
            try:
                for _ in range(100):
                    if active_port.is_file():
                        lines = active_port.read_text().splitlines()
                        if lines and lines[0].isdigit():
                            self.port = int(lines[0])
                            break
                    if self.process.poll() is not None:
                        raise RuntimeError("단부르 전용 브라우저를 시작하지 못했습니다.")
                    await asyncio.sleep(0.1)
                else:
                    raise RuntimeError("단부르 전용 브라우저를 여는 시간이 초과되었습니다.")
                self.client = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20))
                async with self.client.get(f"http://127.0.0.1:{self.port}/json/list") as response:
                    targets = await response.json()
                target = next(item for item in targets if item.get("type") == "page")
                self.target = target["id"]
                self.socket = await self.client.ws_connect(target["webSocketDebuggerUrl"], max_msg_size=16 * 1024 * 1024)
                self.reader = asyncio.create_task(self._read())
                await self.command("Page.enable")
                # Native OS select popups use the hidden window's screen coordinates.
                # Keep their rendering in the page so it shares streamed input coordinates.
                await self.command("Page.addScriptToEvaluateOnNewDocument", {"source": PAGE_CONTROLS})
                await self.command("Emulation.setDeviceMetricsOverride", {
                    "width": self.width, "height": self.height, "deviceScaleFactor": 1, "mobile": False,
                })
                await self.navigate(HOME)
                await self.start_stream()
            except BaseException:
                await self.close()
                raise

    async def close(self):
        self.streaming = False
        self.last_frame = None
        self.pointer_position = None
        self.pointer_revision += 1
        if self.running:
            with contextlib.suppress(Exception):
                await self.command("Browser.close", timeout=2)
        if self.socket:
            await self.socket.close()
            self.socket = None
        if self.reader:
            self.reader.cancel()
            await asyncio.gather(self.reader, return_exceptions=True)
            self.reader = None
        if self.client:
            await self.client.close()
            self.client = None
        if self.process:
            if self.process.poll() is None:
                try:
                    await asyncio.to_thread(self.process.wait, 3)
                except subprocess.TimeoutExpired:
                    with contextlib.suppress(OSError):
                        self.process.terminate()
                    with contextlib.suppress(subprocess.TimeoutExpired):
                        await asyncio.to_thread(self.process.wait, 3)
            self.process = None

    async def snapshot(self):
        if not self.running:
            return {"status": "disconnected", "url": "", "title": "", "width": self.width, "height": self.height}
        page = await self.evaluate("({url:location.href,title:document.title,challenge:!!document.querySelector('#challenge-running, #challenge-stage, #cf-challenge-running')||/just a moment|보안 확인/i.test(document.title),ready:document.readyState,failed:location.protocol==='chrome-error:'||/access denied|error [45]\\d\\d/i.test(document.title)})")
        page = page or {}
        status = "verification" if page.get("challenge") else "error" if page.get("failed") else "loading" if page.get("url") == "about:blank" or page.get("ready") == "loading" else "connected"
        result = {**page, "status": status, "width": self.width, "height": self.height}
        if self.pointer_position:
            with contextlib.suppress(RuntimeError):
                result.update(await self.read_cursor())
        return result

    async def action(self, body):
        action = body.get("action")
        if action == "start":
            await self.start()
        elif action == "stop":
            async with self.lock:
                await self.close()
        elif action == "home":
            await self.navigate(HOME)
        elif action == "reload":
            await self.command("Page.reload")
        elif action == "resize":
            width = max(360, min(1440, int(body.get("width", 1100))))
            height = max(400, min(1000, int(body.get("height", 760))))
            await self.command("Emulation.setDeviceMetricsOverride", {"width": width, "height": height, "deviceScaleFactor": 1, "mobile": False})
            self.width, self.height = width, height
        elif action in ("back", "forward"):
            history = await self.command("Page.getNavigationHistory")
            index = history["currentIndex"] + (-1 if action == "back" else 1)
            if 0 <= index < len(history["entries"]):
                await self.command("Page.navigateToHistoryEntry", {"entryId": history["entries"][index]["id"]})
        elif action in ("pointer", "wheel"):
            x, y = float(body.get("x", 0)), float(body.get("y", 0))
            width, height = float(body.get("width", self.width)), float(body.get("height", self.height))
            if not all(math.isfinite(value) for value in (x, y, width, height)) or width <= 0 or height <= 0:
                raise ValueError("마우스 위치를 확인하세요.")
            # Input may arrive while the next resized frame is still in flight.
            x = max(0, min(self.width - 1, x * self.width / width))
            y = max(0, min(self.height - 1, y * self.height / height))
            params = {"x": x, "y": y}
            if action == "wheel":
                params.update(type="mouseWheel", deltaX=max(-1600, min(1600, float(body.get("dx", 0)))), deltaY=max(-1600, min(1600, float(body.get("dy", 0)))))
            else:
                kind = body.get("kind")
                if kind not in ("mousePressed", "mouseReleased", "mouseMoved"):
                    raise ValueError("마우스 요청 형식을 확인하세요.")
                params.update(type=kind, button="left" if kind != "mouseMoved" or body.get("drag") else "none", buttons=1 if kind == "mousePressed" or body.get("drag") else 0, clickCount=2 if body.get("double") else 1)
            await self.command("Input.dispatchMouseEvent", params)
            self.pointer_position = (x, y)
            self.pointer_revision += 1
            with contextlib.suppress(RuntimeError):
                return {"ok": True, **await self.read_cursor()}
        elif action == "text":
            text = body.get("text")
            if not isinstance(text, str) or len(text) > 10000:
                raise ValueError("입력할 텍스트를 확인하세요.")
            await self.command("Input.insertText", {"text": text})
        elif action == "key":
            key = body.get("key", "")
            if key not in ("Enter", "Tab", "Backspace", "Delete", "Escape", "ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End", "PageUp", "PageDown", "a"):
                raise ValueError("지원하지 않는 키입니다.")
            codes = {"Enter": 13, "Tab": 9, "Backspace": 8, "Delete": 46, "Escape": 27, "ArrowLeft": 37, "ArrowUp": 38, "ArrowRight": 39, "ArrowDown": 40, "Home": 36, "End": 35, "PageUp": 33, "PageDown": 34, "a": 65}
            params = {"key": key, "windowsVirtualKeyCode": codes[key], "modifiers": (2 if body.get("ctrl") else 0) | (8 if body.get("shift") else 0)}
            await self.command("Input.dispatchKeyEvent", {"type": "keyDown" if key == "Enter" else "rawKeyDown", **params, **({"text": "\r"} if key == "Enter" else {})})
            await self.command("Input.dispatchKeyEvent", {"type": "keyUp", **params})
        elif action == "copy":
            return {"text": await self.evaluate("(()=>{const e=document.activeElement;return e&&typeof e.selectionStart==='number'?e.value.slice(e.selectionStart,e.selectionEnd):String(window.getSelection())})()")}
        else:
            raise ValueError("브라우저 작업을 확인하세요.")
        return {"ok": True}


def register_danbooru_browser(app, root, schema):
    manager = DanbooruBrowser(root)

    async def handle(request):
        if request.headers.get("X-Comfy-Settings") != "1":
            raise web.HTTPForbidden()
        if request.headers.get("X-Comfy-Settings-Schema") != schema:
            return web.json_response({"error": "공방 화면을 새로고침해 주세요."}, status=409)
        try:
            if request.method == "POST":
                if request.content_type != "application/json":
                    raise web.HTTPUnsupportedMediaType()
                body = await request.json()
                if not isinstance(body, dict):
                    raise ValueError("브라우저 요청 형식을 확인하세요.")
                async with manager.interaction_lock:
                    result = await manager.action(body)
            else:
                result = await manager.snapshot()
            return web.json_response(result, headers={"Cache-Control": "no-store"})
        except (ValueError, TypeError, OverflowError):
            return web.json_response({"error": "브라우저 요청 값을 확인하세요."}, status=400)
        except (RuntimeError, aiohttp.ClientError, OSError) as exc:
            message = str(exc) if isinstance(exc, RuntimeError) else "단부르 브라우저에 연결하지 못했습니다. 다시 열어 주세요."
            return web.json_response({"error": message}, status=503)

    async def cleanup(_app):
        await manager.close()

    async def stream(request):
        if request.headers.get("Origin") != f"{request.scheme}://{request.host}" or request.query.get("schema") != schema:
            raise web.HTTPForbidden()
        socket = web.WebSocketResponse(heartbeat=20, max_msg_size=1024)
        await socket.prepare(request)
        queue = asyncio.Queue(maxsize=1)
        manager.viewers[socket] = queue

        async def send_frames():
            while not socket.closed:
                await socket.send_json(await queue.get())

        sender = asyncio.create_task(send_frames())
        try:
            async with manager.interaction_lock:
                if manager.last_frame is not None:
                    queue.put_nowait(manager.last_frame)
                await manager.start_stream()
            async for _message in socket:
                pass
        finally:
            manager.viewers.pop(socket, None)
            sender.cancel()
            await asyncio.gather(sender, return_exceptions=True)
            async with manager.interaction_lock:
                if not manager.viewers and manager.running and manager.streaming:
                    with contextlib.suppress(RuntimeError):
                        await manager.command("Page.stopScreencast")
                    manager.streaming = False
        return socket

    async def shutdown(_app):
        await asyncio.gather(*(socket.close(code=1001) for socket in list(manager.viewers)))

    app.router.add_get("/api/illust/danbooru-browser", handle)
    app.router.add_post("/api/illust/danbooru-browser", handle)
    app.router.add_get("/api/illust/danbooru-browser/stream", stream)
    app.on_shutdown.append(shutdown)
    app.on_cleanup.append(cleanup)
    return manager
