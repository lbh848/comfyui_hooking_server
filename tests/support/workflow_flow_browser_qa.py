"""Exercise the real flow UI in local headless Chromium, with no LLM/GPU calls."""
import asyncio
import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

from aiohttp import ClientSession, web

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import illustration_flow as flow


def job(name, kind, params=None):
    item = SimpleNamespace(id=name, label=name, type=kind, params=params or {}, status="pending")
    flow.queue_added(item)
    return item


@flow.llm_call
async def call(name, messages):
    flow.llm_metadata(status="processing", model="QA model", service="local mock")
    return f"{name} 완료 · <script>텍스트로 표시</script>"


class Runner:
    @flow.queue_execution
    async def execute(self, item):
        return await item.handler()


async def prepare_graphs():
    illustration = job("삽화 요청", "illustration_llm_build")
    async def illustrate():
        await call("CALL1", [{"role": "user", "content": "삽화 원문"}])
        return {"success": True}
    illustration.handler = illustrate
    await Runner().execute(illustration)
    video = job("H3 I2V · 5초 영상", "video_prompt_build")
    pending = []
    async def prompt():
        await call("참조 이미지 분석", [{"role": "user", "content": "영상 참조 이미지"}])
        await flow.gather(*(call(f"영상 프롬프트 후보 {i}", [{"role": "user", "content": f"후보 {i} 입력"}]) for i in range(1, 4)))
        await call("최종 후보 선택", [{"role": "user", "content": "후보 비교"}])
        pending.append(job("영상 생성", "video_i2v"))
        return {"success": True}
    video.handler = prompt
    await Runner().execute(video)
    async def render():
        pending.append(job("업스케일 · AVIF 저장", "video_postprocess"))
        return {"success": True}
    pending[0].handler = render
    await Runner().execute(pending[0])
    post = pending[1]
    run, node_id, _ = post._illustration_flow
    flow.update(run, node_id, status="processing", progress=42)
    input_session = "browser-qa-input-session"
    flow.record_video_input_event(
        input_session,
        "session_start",
        input={"reference": "브라우저 QA 원본"},
        output={"instruction": "사람이 작성한 원문"},
    )
    input_edit = job(
        "입력 다듬기",
        "video_instruction_refine",
        {"video_input_session_id": input_session, "instruction": "사람이 작성한 원문"},
    )
    async def refine():
        await call("입력 다듬기 LLM", [{"role": "user", "content": "원문을 다듬기"}])
        return {"success": True, "draft": "보전된 수정문"}
    input_edit.handler = refine
    await Runner().execute(input_edit)
    flow.record_video_input_event(
        input_session,
        "restore_original",
        input={"instruction": "보전된 수정문"},
        output={"instruction": "사람이 작성한 원문"},
    )
    return post


async def main():
    post = await prepare_graphs()
    async def page(request):
        return web.Response(content_type="text/html", text='''<!doctype html><html lang="ko"><meta charset="utf-8">
<style>body{background:#0b1220;color:#e2e8f0;font:14px system-ui}button{font:inherit}</style>
<button id="open" onclick="openWorkflowFlow()">작업 흐름 보기</button><div id="toast"></div>
<script>function showToast(text){document.getElementById('toast').textContent=text}</script>
<script src="/illustration_flow.js"></script></html>''')
    async def script(request):
        return web.FileResponse(ROOT / "frontend" / "illustration_flow.js")
    async def state(request):
        if request.query.get("node"):
            node = flow.detail(request.query.get("run"), request.query["node"])
            return web.json_response({"node": node}, status=200 if node else 404)
        return web.json_response({"flow": flow.snapshot(kind=request.query.get("kind", "illustration"))})
    app = web.Application()
    app.add_routes([web.get("/", page), web.get("/illustration_flow.js", script), web.get("/api/illustration_flow", state)])
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    browser = next((Path(os.environ["LOCALAPPDATA"]) / "ms-playwright").glob("chromium-*/chrome-win64/chrome.exe"))
    process = None
    try:
        with tempfile.TemporaryDirectory(prefix="workflow-flow-browser-") as profile:
            process = subprocess.Popen([
                str(browser), "--headless", "--disable-gpu", "--no-first-run",
                "--remote-debugging-port=0", f"--user-data-dir={profile}", "about:blank",
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW)
            try:
                active_port = Path(profile) / "DevToolsActivePort"
                for _ in range(100):
                    if active_port.exists():
                        break
                    await asyncio.sleep(0.1)
                debug_port = active_port.read_text(encoding="utf-8").splitlines()[0]
                async with ClientSession() as session:
                    async with session.get(f"http://127.0.0.1:{debug_port}/json/list") as response:
                        targets = await response.json()
                    target = next(t for t in targets if t["type"] == "page")
                    async with session.ws_connect(target["webSocketDebuggerUrl"]) as ws:
                        sequence = 0
                        exceptions = []
                        async def cdp(method, params=None):
                            nonlocal sequence
                            sequence += 1
                            await ws.send_json({"id": sequence, "method": method, "params": params or {}})
                            while True:
                                message = await asyncio.wait_for(ws.receive_json(), 15)
                                if message.get("method") == "Runtime.exceptionThrown":
                                    exceptions.append(message["params"])
                                if message.get("id") == sequence:
                                    if "error" in message:
                                        raise AssertionError(message["error"])
                                    return message.get("result", {})
                        async def evaluate(code):
                            result = await cdp("Runtime.evaluate", {
                                "expression": f"(async () => {{ {code} }})()", "awaitPromise": True, "returnByValue": True,
                            })
                            if result.get("exceptionDetails"):
                                raise AssertionError(result["exceptionDetails"])
                            return result.get("result", {}).get("value")
                        await cdp("Runtime.enable")
                        await cdp("Page.enable")
                        await cdp("Emulation.setDeviceMetricsOverride", {"width": 1440, "height": 1000, "deviceScaleFactor": 1, "mobile": False})
                        await cdp("Page.navigate", {"url": f"http://127.0.0.1:{port}/"})
                        for _ in range(100):
                            if await evaluate("return typeof openWorkflowFlow === 'function';"):
                                break
                            await asyncio.sleep(0.05)
                        await evaluate("await openWorkflowFlow();")
                        assert await evaluate("return document.querySelector('#if-title').textContent;") == "작업 흐름 보기"
                        assert await evaluate("return [...document.querySelectorAll('[role=tab]')].map(t => t.textContent);") == ["삽화 흐름 보기", "영상 흐름 보기", "영상 입력 개선"]
                        await evaluate("document.querySelector('#if-tab-video').click(); await refreshIllustrationFlow();")
                        assert await evaluate("return document.querySelector('#if-tab-video').getAttribute('aria-selected');") == "true"
                        assert await evaluate("return document.querySelectorAll('.if-node').length;") == 8
                        assert await evaluate("return document.querySelectorAll('.if-edges path').length;") == 9
                        assert await evaluate("return document.querySelector('#if-stop').hidden && document.querySelector('#if-developer-mode').hidden;")
                        assert await evaluate("return [...document.querySelectorAll('.if-node-state')].some(n => n.textContent.includes('42%'));")
                        await evaluate("document.querySelector('.if-port[data-node-id=\"영상 생성\"]').click();")
                        for _ in range(100):
                            if await evaluate("return document.querySelector('.if-detail h2').textContent === '영상 생성';"):
                                break
                            await asyncio.sleep(0.05)
                        assert await evaluate("return document.querySelector('.if-detail-body').textContent.includes('success');")
                        await evaluate("document.querySelector('.if-detail').close(); document.querySelector('#if-tab-video_input').click();")
                        assert await evaluate("return [...document.querySelectorAll('.if-node-title')].map(n=>n.textContent);") == ["영상 입력 시작", "입력 다듬기", "입력 다듬기 LLM", "원문으로 되돌리기"]
                        assert await evaluate("return [...document.querySelectorAll('.if-node')].some(n=>n.dataset.executor==='human');")
                        await evaluate("document.querySelector('#if-tab-illustration').click();")
                        assert await evaluate("return document.querySelectorAll('.if-node').length;") == 3
                        await evaluate("document.querySelector('#if-tab-illustration').dispatchEvent(new KeyboardEvent('keydown', {key:'ArrowRight', bubbles:true}));")
                        assert await evaluate("return document.activeElement.id;") == "if-tab-video"
                        await evaluate("const buttons=[...document.querySelectorAll('.if-actions button')]; for(let i=0;i<3;i++) buttons.find(b=>b.textContent==='−').click();")
                        zoom = await evaluate("return document.querySelector('#if-zoom-reset').textContent;")
                        await evaluate("document.querySelector('#if-tab-illustration').click(); document.querySelector('#if-tab-video').click();")
                        assert await evaluate("return document.querySelector('#if-zoom-reset').textContent;") == zoom
                        await evaluate("const r=await fetch('/api/illustration_flow');const {flow}=await r.json();receiveIllustrationFlow({...flow,id:'new-illustration',created_at:flow.created_at+100,revision:1},true);")
                        assert await evaluate("return document.querySelector('#if-tab-video').getAttribute('aria-selected');") == "true"
                        screenshot = await cdp("Page.captureScreenshot", {"format": "png"})
                        artifact = ROOT / "요구사항" / f"workflow_flow_qa_{time.time_ns()}.png"
                        artifact.parent.mkdir(exist_ok=True)
                        artifact.write_bytes(base64.b64decode(screenshot["data"]))
                        await cdp("Emulation.setDeviceMetricsOverride", {"width": 390, "height": 844, "deviceScaleFactor": 1, "mobile": False})
                        assert await evaluate("const r=document.querySelector('#illustration-flow-modal').getBoundingClientRect();return r.left>=0 && r.right<=innerWidth && r.bottom<=innerHeight;")
                        assert await evaluate("return document.querySelector('.if-footer').getBoundingClientRect().bottom <= document.querySelector('#illustration-flow-modal').getBoundingClientRect().bottom;")
                        async def complete():
                            return {"success": True, "backup_name": "qa.avif"}
                        post.handler = complete
                        await Runner().execute(post)
                        await evaluate("await refreshIllustrationFlow();")
                        assert await evaluate("return [...document.querySelectorAll('.if-node-title')].some(n=>n.textContent==='결과 반환');")
                        await evaluate("document.querySelector('#illustration-flow-modal').close(); await new Promise(resolve=>setTimeout(resolve,50));")
                        assert await evaluate("return document.querySelector('.if-layer-backdrop').hidden;")
                        assert not exceptions, exceptions
                        print(json.dumps({"browser_qa": "passed", "screenshot": str(artifact), "browser_pid": process.pid}, ensure_ascii=False))
            finally:
                process.terminate()
                process.wait(timeout=10)
                print(f"[FLOW_QA] Browser stopped: pid={process.pid}, exit={process.returncode}")
    finally:
        await runner.cleanup()
        print(f"[FLOW_QA] Preview server stopped: port={port}")


if __name__ == "__main__":
    asyncio.run(main())
