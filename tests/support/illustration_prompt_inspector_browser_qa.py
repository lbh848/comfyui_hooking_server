"""Isolated real-browser checks; no production API, LLM, GPU or saved data writes."""
import asyncio
import base64
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

from aiohttp import ClientSession, web


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from modes import illustration_context_pipeline as pipeline
from modes.illustration_prompt_structure import build_prompt_structure


def function_source(html, name):
    match = re.search(rf"^        (?:async )?function {name}\(", html, re.MULTILINE)
    if match is None:
        raise ValueError(f"UI function missing: {name}")
    end = html.index("\n        }", match.start()) + len("\n        }")
    return html[match.start():end]


async def main():
    html = (ROOT / "frontend/index.html").read_text(encoding="utf-8")
    css = re.search(r"<style>(.*?)</style>", html, re.DOTALL).group(1)
    helpers = "\n".join(function_source(html, name) for name in ("openIllustLogModal", "switchIllustModalTab", "switchIllustCallTab"))
    page = "<html><head><meta charset='utf-8'><style>" + css + "</style></head><body><script src='/illustration_prompt_inspector.js'></script><script>" + """
        window.instructionLoads = 0;
        function refreshIllustLogs() {}
        function loadLighbdPrompts() { window.instructionLoads++; }
        function loadIllustrationContextToggles() {}
        function loadIllustrationChatHistories() {}
    """ + helpers + "</script></body></html>"
    prompts = pipeline.load_prompt_files()
    prompts['call2_select'] += '\n<script>window.unexpected=true</script>\nEND OF FIXED INSTRUCTION'
    blueprint = build_prompt_structure(prompts)
    requests = []
    api_failure = False

    async def serve_page(request):
        return web.Response(text=page, content_type="text/html")

    async def serve_script(request):
        return web.FileResponse(ROOT / "frontend/illustration_prompt_inspector.js")

    async def serve_structure(request):
        requests.append(request.path)
        if api_failure:
            return web.json_response({"error": "fixture failure"}, status=503)
        return web.json_response(blueprint)

    async def forbidden_api(request):
        requests.append(request.path)
        raise AssertionError(f"Unexpected API: {request.method} {request.path}")

    app = web.Application()
    app.add_routes([web.get("/", serve_page), web.get("/illustration_prompt_inspector.js", serve_script),
                    web.get("/api/illustration_context/prompt-structure", serve_structure),
                    web.route("*", "/api/{path:.*}", forbidden_api)])
    runner = web.AppRunner(app)
    await runner.setup()
    process = None
    try:
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        browsers = list((Path(os.environ["LOCALAPPDATA"]) / "ms-playwright").glob("chromium-*/chrome-win64/chrome.exe"))
        if not browsers:
            raise RuntimeError("Installed Chromium unavailable; real-browser QA cannot run")
        with tempfile.TemporaryDirectory(prefix="illust-input-browser-") as profile:
            process = subprocess.Popen([str(browsers[-1]), "--headless", "--disable-gpu", "--no-first-run", "--remote-debugging-port=0", f"--user-data-dir={profile}", "about:blank"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
            try:
                active_port = Path(profile) / "DevToolsActivePort"
                for _ in range(100):
                    if active_port.exists():
                        break
                    await asyncio.sleep(0.1)
                debug_port = active_port.read_text(encoding="utf-8").splitlines()[0]
                async with ClientSession() as session:
                    async with session.get(f"http://127.0.0.1:{debug_port}/json/list") as response:
                        target = next(item for item in await response.json() if item["type"] == "page")
                    async with session.ws_connect(target["webSocketDebuggerUrl"]) as ws:
                        sequence = 0
                        exceptions = []

                        async def cdp(method, params=None):
                            nonlocal sequence
                            sequence += 1
                            await ws.send_json({"id": sequence, "method": method, "params": params or {}})
                            while True:
                                message = await asyncio.wait_for(ws.receive_json(), 20)
                                if message.get("method") == "Runtime.exceptionThrown":
                                    exceptions.append(message["params"])
                                if message.get("id") == sequence:
                                    if "error" in message:
                                        raise AssertionError(message["error"])
                                    return message.get("result", {})

                        async def evaluate(code):
                            result = await cdp("Runtime.evaluate", {"expression": "(async()=>{" + code + "})()", "awaitPromise": True, "returnByValue": True})
                            if "exceptionDetails" in result:
                                raise AssertionError(result["exceptionDetails"])
                            return result.get("result", {}).get("value")

                        async def wait_for(condition):
                            for _ in range(100):
                                if await evaluate("return " + condition + ";"):
                                    return
                                await asyncio.sleep(0.05)
                            raise AssertionError(f"Browser condition timed out: {condition}")

                        async def check_flow_alignment(width):
                            await cdp("Emulation.setDeviceMetricsOverride", {"width": width, "height": 960, "deviceScaleFactor": 1, "mobile": False})
                            geometry = await evaluate("""
                                await new Promise(resolve=>requestAnimationFrame(resolve));
                                const bounds = element => {
                                    const rect = element.getBoundingClientRect();
                                    return {top:rect.top, bottom:rect.bottom, left:rect.left, right:rect.right, height:rect.height};
                                };
                                return [...document.querySelectorAll('.illust-structure-lane')].map(lane=>({
                                    lane:bounds(lane), label:bounds(lane.querySelector('.illust-structure-lane-label')),
                                    nodes:[...lane.querySelectorAll('.illust-structure-node')].map(bounds)
                                }));
                            """)
                            rows = []
                            for lane in geometry:
                                if not rows or abs(rows[-1][0]['lane']['top'] - lane['lane']['top']) > 1:
                                    rows.append([])
                                rows[-1].append(lane)
                                previous_bottom = lane['label']['bottom']
                                for node in lane['nodes']:
                                    assert node['top'] >= previous_bottom, (width, lane)
                                    assert node['left'] >= lane['lane']['left'] - 1 and node['right'] <= lane['lane']['right'] + 1, (width, lane)
                                    previous_bottom = node['bottom']
                            for row in rows:
                                for index in range(max(len(lane['nodes']) for lane in row)):
                                    nodes = [lane['nodes'][index] for lane in row if len(lane['nodes']) > index]
                                    assert max(node['top'] for node in nodes) - min(node['top'] for node in nodes) < 1, (width, index, nodes)
                                    assert max(node['height'] for node in nodes) - min(node['height'] for node in nodes) < 1, (width, index, nodes)
                            assert await evaluate("const root=document.querySelector('#illust-assembled-inputs');return root.scrollWidth<=root.clientWidth;"), width
                            print(f"[ILLUST_INPUT_QA] Flow rows aligned: width={width}, rows={len(rows)}")

                        await cdp("Runtime.enable")
                        await cdp("Emulation.setDeviceMetricsOverride", {"width": 1280, "height": 960, "deviceScaleFactor": 1, "mobile": False})
                        await cdp("Page.navigate", {"url": f"http://127.0.0.1:{port}/"})
                        await wait_for("typeof openIllustLogModal === 'function'")
                        await evaluate("openIllustLogModal();")
                        await wait_for("document.querySelectorAll('.illust-input-message pre').length===2")
                        assert await evaluate("return document.querySelector('#illust-tab-llm').getAttribute('aria-selected')==='true' && !document.querySelector('#illust-instruction-editor').open && window.instructionLoads===0;")
                        assert await evaluate("return document.querySelectorAll('.illust-structure-node').length===12 && document.querySelectorAll('.illust-structure-lane').length===7;")
                        for width in (1440, 1280, 1000, 760, 650, 390, 320):
                            await check_flow_alignment(width)
                        await evaluate("""
                            window.flowLabels = [...document.querySelectorAll('.illust-structure-lane-label')].map(label=>label.textContent);
                            document.querySelectorAll('.illust-structure-lane-label')[1].textContent = '설명이 길어져 여러 줄로 바뀌는 단계: 원문과 등록 자료를 준비하고 현재 문맥에 맞춰 다음 단계로 전달합니다.';
                        """)
                        for width in (1280, 760, 390):
                            await check_flow_alignment(width)
                        await evaluate("document.querySelectorAll('.illust-structure-lane-label').forEach((label,index)=>label.textContent=window.flowLabels[index]);")
                        await check_flow_alignment(1280)
                        assert await evaluate("return !document.querySelector('#illust-input-call') && !document.querySelector('#illust-input-source') && !window.unexpected;")
                        assert await evaluate("return document.querySelector('[data-role=system] h4').textContent.includes('시스템 프롬프트 (SYSTEM)') && document.querySelector('[data-role=user] h4').textContent.includes('유저 프롬프트 (USER)');")
                        assert await evaluate("return [...document.querySelectorAll('.illust-input-message mark')].every(el=>el.textContent.startsWith('{')&&el.textContent.endsWith('}')&&getComputedStyle(el).backgroundColor==='rgb(23, 70, 120)');")
                        assert await evaluate("return document.querySelector('[data-role=user] pre').textContent.includes('{과거 문맥 입력}') && document.querySelector('[data-role=user] pre').textContent.includes('Return only') && !document.querySelector('#illust-assembled-inputs').textContent.includes('_input_');")
                        assert await evaluate("return document.querySelector('[data-role=system] pre').textContent.endsWith('END OF FIXED INSTRUCTION') && !document.querySelector('[data-role=system] pre').innerHTML.includes('<script>');")
                        for stage in blueprint['stages']:
                            await evaluate("document.querySelector('[data-key=" + stage['key'] + "]').click();")
                            texts = await evaluate("return [...document.querySelectorAll('.illust-input-message pre')].map(el=>el.textContent);")
                            assert texts == [message['content'] for message in stage['messages']], stage['key']
                        await evaluate("document.querySelector('[data-key=detail]').click();")
                        assert await evaluate("""
                            const blocks=[...document.querySelectorAll('.illust-structure-block')];
                            return blocks.length===4 && blocks.every(block=>block.dataset.role==='SYSTEM') &&
                                blocks[0].dataset.source==='prompts/lighbd/jailbreak.txt' &&
                                blocks[1].textContent.includes('{lb-xnai.lb.extra 시스템 프롬프트}') &&
                                blocks[2].dataset.source==='prompts/lighbd/explicit.txt' &&
                                blocks[3].dataset.source==='prompts/lighbd/detail.txt';
                        """)
                        await evaluate("document.querySelectorAll('.illust-structure-block').forEach(block=>block.open=true);")
                        detail_stage = next(stage for stage in blueprint['stages'] if stage['key'] == 'detail')
                        assert await evaluate("return [...document.querySelectorAll('.illust-structure-block pre')].map(pre=>pre.textContent);") == [part['content'] for part in detail_stage['system_parts']]
                        assert await evaluate("return document.querySelector('.illust-structure-notes').textContent.includes('call2_common') && !document.querySelector('[data-role=system] pre').textContent.includes('Convert each authorized visual moment');")
                        await check_flow_alignment(390)
                        await check_flow_alignment(1280)
                        await evaluate("document.querySelector('[data-key=call3]').click(); const select=document.querySelector('.illust-structure-variant select');select.value='manga';select.dispatchEvent(new Event('change'));")
                        assert await evaluate("return document.querySelector('[data-role=system] pre').textContent.includes('{만화 NSFW 설정}');")
                        await evaluate("document.querySelector('[data-key=keyvis]').click();")
                        assert await evaluate("return document.querySelectorAll('.illust-structure-block[data-role=SYSTEM]').length===5 && document.querySelector('.illust-structure-block[data-role=USER]').textContent.includes('{lb-xnai.lb.extra 시스템 프롬프트}');")
                        assert await evaluate("return document.querySelector('[data-role=system] pre').textContent.includes('[그 외 조건]') && document.querySelector('[data-role=system] pre').textContent.includes('Prioritize character state');")
                        await evaluate("document.querySelector('[data-key=call1]').click();")
                        assert await evaluate("return document.querySelector('.illust-structure-purpose').textContent.includes('과거 문맥') && document.querySelector('.illust-structure-notes').textContent.includes('저장 이력이 없어도');")
                        assert await evaluate("const block=document.querySelector('.illust-structure-block[data-source=\"prompts/lighbd/enhance.txt\"]');block.open=true;return block.querySelector('pre').textContent.trim()===document.querySelector('.illust-input-message[data-role=system] pre').textContent.trim();")
                        assert await evaluate("return document.querySelector('.illust-input-message[data-role=user] pre').textContent.includes('RECONSTRUCT OUTFIT AND HAIRSTYLE IMMEDIATELY BEFORE CURRENT');")
                        api_failure = True
                        await evaluate("await refreshIllustrationPromptInspector();")
                        assert await evaluate("return document.querySelector('.illust-input-error').textContent.includes('503');")
                        api_failure = False
                        prompts['call2_select'] += '\nRELOADED FIXED INSTRUCTION'
                        blueprint = build_prompt_structure(prompts)
                        await evaluate("await refreshIllustrationPromptInspector(); document.querySelector('[data-key=select]').click();")
                        assert await evaluate("return document.querySelector('[data-role=system] pre').textContent.endsWith('RELOADED FIXED INSTRUCTION');")
                        assert await evaluate("return document.querySelector('#illust-tab-content-llm').textContent.includes('.txt')===false;")
                        await evaluate("document.querySelector('#illust-instruction-editor').open=true;await new Promise(resolve=>setTimeout(resolve,30));document.querySelector('#illust-instruction-editor').open=false;await new Promise(resolve=>setTimeout(resolve,30));document.querySelector('#illust-instruction-editor').open=true;await new Promise(resolve=>setTimeout(resolve,30));")
                        assert await evaluate("return window.instructionLoads===1;")
                        await evaluate("document.querySelector('#illust-instruction-editor').open=false;document.querySelector('[data-key=detail]').click();document.querySelector('.modal-body').scrollTop=0;")
                        assert requests and set(requests) == {'/api/illustration_context/prompt-structure'}, requests
                        await cdp("Page.enable")
                        artifact = ROOT / "tests/artifacts/illustration_prompt_inspector.png"
                        artifact.parent.mkdir(exist_ok=True)
                        shot = await cdp("Page.captureScreenshot", {"format": "png"})
                        artifact.write_bytes(base64.b64decode(shot["data"]))
                        await cdp("Emulation.setDeviceMetricsOverride", {"width": 390, "height": 844, "deviceScaleFactor": 1, "mobile": False})
                        assert await evaluate("const rect=document.querySelector('.illust-prompt-modal-content').getBoundingClientRect();return rect.left>=0 && rect.right<=innerWidth && document.querySelector('#illust-assembled-inputs').scrollWidth<=document.querySelector('#illust-assembled-inputs').clientWidth;")
                        assert not exceptions, exceptions
                        print(json.dumps({"browser_qa": "passed", "checks": "all 12 stages without generation or history, Korean system/user roles, blue named placeholders, shared assembly text, conditional branches, mode variants, escaping, reload, API failure, lazy editor, mobile width", "screenshot": str(artifact)}, ensure_ascii=False))
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=10)
                print(f"[ILLUST_INPUT_QA] Browser stopped: pid={process.pid}, exit={process.returncode}")
    finally:
        await runner.cleanup()
        print("[ILLUST_INPUT_QA] Isolated API server stopped")


if __name__ == "__main__":
    asyncio.run(main())
