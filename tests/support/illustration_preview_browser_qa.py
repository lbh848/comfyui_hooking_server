"""Exercise the real preview in Chromium without production API or data writes."""
import asyncio
import base64
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
import traceback

from aiohttp import ClientSession, web

ROOT = Path(__file__).resolve().parents[2]


async def main():
    html = (ROOT / "frontend/index.html").read_text(encoding="utf-8")
    css = re.search(r"<style>(.*?)</style>", html, re.DOTALL).group(1)
    start = html.index("        function renderIllustrationNodePreview()")
    render = html[start:html.index("\n        }", start) + len("\n        }")]
    fallback_start = html.index("        function llmRouteResolvedFallback(")
    fallback = html[fallback_start:html.index("\n        }", fallback_start) + len("\n        }")]
    saved = json.loads((ROOT / "config.json").read_text(encoding="utf-8-sig"))
    # Keep only the public routing and limits needed by the preview.
    saved_config = {key: value for key, value in saved.items()
                    if key == "llm_routing" or key.startswith("llm_max_concurrency") or re.fullmatch(r"llm_model\d*", key)}
    page = "<html lang='ko'><meta charset='utf-8'><style>" + css + "</style>" + """
    <body style="background:var(--bg);color:var(--text);padding:20px">
    <div class="modal visible" style="display:block;position:static">
    <div id="illust-tab-content-toggles" class="illust-settings">
    <section class="illust-flow-preview">
    <h3>노드 배치 미리보기 · 실제 LLM 제한</h3>
    <div class="illust-preview-toolbar">
    <button data-illust-preview-play>흐름 재생</button><button data-illust-preview-reset>처음부터</button>
    <select data-illust-preview-speed><option value="1">1×</option><option value="2">2×</option></select>
    <input type="range" min="0" step="0.01" data-illust-preview-progress>
    <span data-illust-preview-status></span></div>
    <div class="illust-flow-preview-viewport"><div id="illust-node-preview" class="illust-flow-preview-grid"></div></div>
    </section></div></div>
    <script src="/illustration_preview.js"></script><script src="/llm_thinking_check.js"></script><script>
    const LLM_SLOTS = [1,2,3,4,5,6,7,8,9,10];
    let configReads = 0;
    const loadCurrentConfig = async force => {if (!force) throw Error('Expected saved config reload'); configReads++; return currentConfig;};
    const escapeHtml = value => String(value ?? '').replaceAll('&','&amp;').replaceAll('<','&lt;')
        .replaceAll('>','&gt;').replaceAll('"','&quot;');
    let _illustrationSettingsActiveGroup = 'call2_detail';
    let _illustrationContextToggles = {illustration_output_mode:'illustration',
        call1_backtranslate_enabled:true,call1_backtranslate_max_concurrency:3,
        call2_parallel_enabled:true,call2_parallel_max_concurrency:3,
        call2_detail_parallel_enabled:true,call2_detail_parallel_max_concurrency:5,
        key_visual:true,speak_enabled:true,prompt_format:'v3',character_limit:2,multi_char_mask_enabled:true};
    let currentConfig = """ + json.dumps(saved_config) + ";\n" + fallback + "\n" + render + "\nrenderIllustrationNodePreview();</script></body></html>"
    requests = []
    checks = []

    async def serve_page(request):
        return web.Response(text=page, content_type="text/html")

    async def serve_script(request):
        return web.FileResponse(ROOT / "frontend/illustration_preview.js")

    async def serve_thinking_script(request):
        return web.FileResponse(ROOT / "frontend/llm_thinking_check.js")

    async def thinking_check(request):
        body = await request.json()
        checks.append(body)
        scenario = {
            "llm3": {"thinking_status": "observed", "reasoning_tokens": 17},
            "llm5": {"thinking_status": "zero", "reasoning_tokens": 0},
            "llm7": {"thinking_status": "unknown", "reasoning_chars": 0},
            "llm9": {"thinking_status": "observed", "reasoning_chars": 11},
        }
        if body["target"] == "llm10":
            event, data = "error", {"error": "제공자 연결 실패"}
        else:
            event, data = "done", {"text": "391", "elapsed": 0.12, **scenario[body["target"]]}
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        payload = f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")
        for index in range(0, len(payload), 7):
            await response.write(payload[index:index + 7])
        await response.write_eof()
        return response

    async def forbidden_api(request):
        requests.append(request.path)
        print(f"[ILLUST_PREVIEW_QA] Unexpected API: {request.method} {request.path}")
        raise AssertionError(request.path)

    app = web.Application()
    app.add_routes([web.get("/", serve_page), web.get("/illustration_preview.js", serve_script),
                    web.get("/llm_thinking_check.js", serve_thinking_script),
                    web.post("/api/llm/test_stream", thinking_check),
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
            raise RuntimeError("Installed Chromium unavailable")
        with tempfile.TemporaryDirectory(prefix="illust-preview-llm-") as profile:
            process = subprocess.Popen([str(browsers[-1]), "--headless", "--disable-gpu", "--no-first-run",
                "--remote-debugging-port=0", f"--user-data-dir={profile}", "about:blank"],
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
                            result = await cdp("Runtime.evaluate", {"expression": "(async()=>{" + code + "})()",
                                "awaitPromise": True, "returnByValue": True})
                            if "exceptionDetails" in result:
                                raise AssertionError(result["exceptionDetails"])
                            return result.get("result", {}).get("value")

                        await cdp("Runtime.enable")
                        await cdp("Emulation.setDeviceMetricsOverride", {"width": 1440, "height": 1100,
                            "deviceScaleFactor": 1, "mobile": False})
                        await cdp("Page.navigate", {"url": f"http://127.0.0.1:{port}/"})
                        for _ in range(100):
                            if await evaluate("return !!document.querySelector('.illust-preview-state');"):
                                break
                            await asyncio.sleep(0.05)
                        routes = await evaluate("return [...document.querySelectorAll('.illust-preview-llm')].map(n=>n.textContent);")
                        assert any("LLM3" in route for route in routes), routes
                        print("[ILLUST_PREVIEW_QA] Saved routing rendered:", sorted(set(routes)))
                        await evaluate("""
                            currentConfig={llm_max_concurrency3:1,llm_routing:Object.fromEntries(
                                Object.keys(currentConfig.llm_routing).map(key=>[key,{primary:'llm3'}]))};
                            renderIllustrationNodePreview();
                        """)
                        single = await evaluate("""
                            const slider=document.querySelector('[data-illust-preview-progress]');
                            const stages=[...document.querySelectorAll('.illust-flow-preview-node')];
                            let peak=0;
                            for(let t=0;t<Number(slider.max);t+=0.05){
                                slider.value=t;slider.oninput();
                                peak=Math.max(peak,stages.filter(node=>node.querySelector('.illust-preview-llm')
                                    &&node.dataset.simulationState==='processing').length);
                            }
                            return {peak,duration:Number(slider.max),plans:stages.filter(n=>n.dataset.illustPreviewNode==='CALL2-PLAN').length};
                        """)
                        assert single["peak"] == 1 and single["plans"] == 3, single
                        print("[ILLUST_PREVIEW_QA] SINGLE: three batches retained, one LLM node active")
                        parallel = await evaluate("""
                            currentConfig.llm_max_concurrency3=2;renderIllustrationNodePreview();
                            const slider=document.querySelector('[data-illust-preview-progress]');
                            const reset=Number(slider.value)===0;
                            const stages=[...document.querySelectorAll('.illust-flow-preview-node')];
                            let peak=0,front=false;
                            for(let t=0;t<Number(slider.max);t+=0.05){slider.value=t;slider.oninput();
                                const active=stages.filter(node=>node.querySelector('.illust-preview-llm')&&node.dataset.simulationState==='processing');
                                peak=Math.max(peak,active.length);
                                front ||= active.some(n=>n.dataset.illustPreviewNode==='CALL1')&&active.some(n=>n.dataset.illustPreviewNode==='CALL2-SELECT');
                            }
                            return {peak,front,reset,duration:Number(slider.max)};
                        """)
                        assert parallel["peak"] == 2 and parallel["front"] and parallel["reset"], parallel
                        assert parallel["duration"] < single["duration"]
                        shared_three = await evaluate("""
                            currentConfig.llm_max_concurrency3=3;renderIllustrationNodePreview();
                            const slider=document.querySelector('[data-illust-preview-progress]');
                            const group=document.querySelector('[data-illust-preview-group=detail_keyvis]');
                            const nodes=[...group.querySelectorAll('[data-illust-preview-node]')];
                            const all=[...document.querySelectorAll('[data-illust-preview-node]')];
                            const call3=all.find(node=>node.dataset.illustPreviewNode==='CALL3');
                            let peak=0,sharedWaiting=false,dialogueOverlap=false;
                            for(let t=0;t<Number(slider.max);t+=0.05){
                                slider.value=t;slider.oninput();
                                const active=nodes.filter(node=>node.dataset.simulationState==='processing').length;
                                peak=Math.max(peak,active);
                                const text=group.querySelector(':scope > .illust-preview-group-state').textContent;
                                sharedWaiting ||= text.includes('LLM3 공유 상한 3개 · 처리 중 3개 · 대기 3개 · 완료 0/6');
                                dialogueOverlap ||= call3.dataset.simulationState==='processing' && all.some(node=>
                                    node.dataset.illustPreviewNode==='이미지 생성'&&node.dataset.simulationState==='processing');
                            }
                            return {peak,sharedWaiting,dialogueOverlap,total:nodes.length,
                                step5:call3.closest('.illust-flow-preview-step').getAttribute('aria-label')==='STEP 5'};
                        """)
                        assert shared_three == {"peak": 3, "sharedWaiting": True, "dialogueOverlap": True,
                                                "total": 6, "step5": True}, shared_three
                        print("[ILLUST_PREVIEW_QA] Six DETAIL/KEYVIS batches: shared peak 3, waiting 3; STEP 5 CALL3 overlaps image generation")
                        independent = await evaluate("""
                            currentConfig.llm_max_concurrency3=1;
                            currentConfig.llm_max_concurrency5=1;
                            currentConfig.llm_routing.illustration_call1={primary:'llm5'};
                            renderIllustrationNodePreview();
                            const slider=document.querySelector('[data-illust-preview-progress]');
                            const stages=[...document.querySelectorAll('.illust-flow-preview-node')];
                            for(let t=0;t<Number(slider.max);t+=0.05){slider.value=t;slider.oninput();
                                const active=stages.filter(n=>n.dataset.simulationState==='processing').map(n=>n.dataset.illustPreviewNode);
                                if(active.includes('CALL1')&&active.includes('CALL2-SELECT'))return true;
                            }
                            return false;
                        """)
                        assert independent
                        await evaluate("""
                            document.querySelector('[data-illust-preview-reset]').click();
                            document.querySelector('[data-illust-preview-play]').click();
                            await new Promise(resolve=>setTimeout(resolve,100));
                        """)
                        assert await evaluate("return document.querySelector('#illust-node-preview').dataset.simulationPlaying==='true';")
                        await evaluate("document.querySelector('[data-illust-preview-play]').click();")
                        assert await evaluate("return document.querySelector('#illust-node-preview').dataset.simulationPlaying==='false';")
                        assert not requests, requests
                        assert not exceptions, exceptions
                        assert not checks, checks
                        print("[ILLUST_PREVIEW_QA] Preview PASS: shared limits, waiting counts, independent slots, reset and playback; no automatic API calls")
                        diagnostic = await evaluate("""
                            currentConfig.llm_routing.illustration_call1={primary:'llm3',fallback_target:'llm5'};
                            currentConfig.llm_routing.illustration_call2_select={primary:'llm7',fallback_target:'llm9'};
                            currentConfig.llm_routing.illustration_call2_plan={primary:'llm10'};
                            for(const slot of [1,5,7,9,10])currentConfig[`llm_model${slot===1?'':slot}`]=`mock-model-${slot}`;
                            currentConfig.llm_model3='<script>must render as text</script>';
                            renderIllustrationNodePreview();
                            const panel=document.querySelector('[data-illust-thinking-checks]');
                            const slots=[...panel.querySelectorAll('[data-thinking-slot]')].map(n=>n.dataset.thinkingSlot);
                            panel.querySelector('button').click();
                            for(let n=0;n<100;n++){
                                await new Promise(resolve=>setTimeout(resolve,30));
                                if([...panel.querySelectorAll('[data-thinking-status]')].every(n=>!['idle','checking'].includes(n.dataset.thinkingStatus)))break;
                            }
                            return {slots,reads:configReads,
                                states:[...panel.querySelectorAll('[data-thinking-status]')].map(n=>({status:n.dataset.thinkingStatus,text:n.textContent})),
                                safe:panel.querySelectorAll('script').length===0,
                                placed:document.querySelector('.illust-flow-preview').nextElementSibling===panel};
                        """)
                        assert diagnostic["slots"] == ["llm3", "llm5", "llm7", "llm9", "llm10"], diagnostic
                        assert [item["status"] for item in diagnostic["states"]] == ["observed", "zero", "unknown", "observed", "error"], diagnostic
                        assert diagnostic["safe"] and diagnostic["placed"] and diagnostic["reads"] == 1, diagnostic
                        assert all(check == {"target": slot, "thinking_check": True}
                                   for check, slot in zip(checks, diagnostic["slots"], strict=True))
                        badges = await evaluate("""
                            currentConfig.llm_routing.illustration_call3={primary:'llm5',fallback_target:'llm7'};
                            currentConfig.llm_routing.illustration_profile_resolve={primary:'llm1',fallback_target:'llm5'};
                            currentConfig.llm_routing.illustration_call2_keyvis={primary:'llm5',fallback_target:'none'};
                            renderIllustrationNodePreview();
                            const color=name=>document.querySelector(`[data-illust-preview-node="${name}"] .illust-thinking-node-badge`)?.dataset.thinkingHealth;
                            return {mainThink:color('CALL1'),fallbackThink:color('CALL2-SELECT'),
                                normal:color('CALL3'),unexamined:color('PROFILE-RESOLVE'),missingFallback:color('CALL2-KEYVIS'),
                                failed:color('CALL2-PLAN'),details:[...document.querySelectorAll('[data-illust-preview-node="CALL2-DETAIL"] .illust-thinking-node-badge')].map(n=>n.dataset.thinkingHealth),
                                noImageBadge:!document.querySelector('[data-illust-preview-node="이미지 생성"] .illust-thinking-node-badge'),
                                noDisabledBadge:!document.querySelector('[data-enabled="false"] .illust-thinking-node-badge')};
                        """)
                        assert badges == {"mainThink": "red", "fallbackThink": "yellow", "normal": "green",
                            "unexamined": "gray", "missingFallback": "red", "failed": "red", "details": ["red"] * 5,
                            "noImageBadge": True, "noDisabledBadge": True}, badges
                        bounds = await evaluate("""
                            const badge=document.querySelector('[data-illust-preview-node="CALL1"] .illust-thinking-node-badge');
                            badge.scrollIntoView({block:'center'});
                            const a=badge.getBoundingClientRect(),b=badge.parentElement.getBoundingClientRect();
                            return {x:a.left+a.width/2,y:a.top+a.height/2,left:a.left,nodeLeft:b.left};
                        """)
                        assert bounds["left"] < bounds["nodeLeft"] and bounds["left"] >= 0, bounds
                        await cdp("Input.dispatchMouseEvent", {"type": "mouseMoved", "x": bounds["x"], "y": bounds["y"]})
                        hover = await evaluate("""
                            const tip=document.querySelector('.illust-thinking-node-tooltip');
                            return {visible:!!tip&&!tip.hidden,text:tip?.textContent,safe:tip?.querySelectorAll('script').length===0};
                        """)
                        assert hover["visible"] and "추론 17토큰" in hover["text"] and "폴백" in hover["text"] and hover["safe"], hover
                        screenshot = await cdp("Page.captureScreenshot", {"format": "png"})
                        badge_artifact = ROOT / "tests/artifacts" / f"thinking_badges_browser_{int(time.time())}.png"
                        badge_artifact.parent.mkdir(parents=True, exist_ok=True)
                        badge_artifact.write_bytes(base64.b64decode(screenshot["data"]))
                        print(f"[THINK_BADGE_QA] Screenshot: {badge_artifact}")
                        await cdp("Input.dispatchMouseEvent", {"type": "mouseMoved", "x": 5, "y": 5})
                        assert await evaluate("return document.querySelector('.illust-thinking-node-tooltip').hidden;")
                        await evaluate("document.querySelector('[data-illust-preview-node=CALL3] .illust-thinking-node-badge').focus();")
                        assert await evaluate("return !document.querySelector('.illust-thinking-node-tooltip').hidden;")
                        await evaluate("document.dispatchEvent(new KeyboardEvent('keydown',{key:'Escape'}));")
                        assert await evaluate("return document.querySelector('.illust-thinking-node-tooltip').hidden;")
                        print("[THINK_BADGE_QA] PASS: all four colors, main/fallback precedence, missing fallback, unknown-as-unused, batches, left placement, real hover, focus and Escape")
                        await evaluate("document.querySelector('[data-illust-thinking-checks]').scrollIntoView({block:'center'});")
                        screenshot = await cdp("Page.captureScreenshot", {"format": "png"})
                        artifact = ROOT / "tests/artifacts" / f"thinking_check_browser_{int(time.time())}.png"
                        artifact.parent.mkdir(parents=True, exist_ok=True)
                        artifact.write_bytes(base64.b64decode(screenshot["data"]))
                        print(f"[THINK_CHECK_QA] Screenshot: {artifact}")
                        assert await evaluate("""
                            currentConfig.llm_model3='changed connection';renderIllustrationNodePreview();
                            return document.querySelector('[data-thinking-slot=llm3] [data-thinking-status]').dataset.thinkingStatus==='idle'
                                &&document.querySelector('[data-illust-preview-node=CALL1] .illust-thinking-node-badge').dataset.thinkingHealth==='gray';
                        """)
                        # One click reloads saved config and only checks the chosen slot.
                        await evaluate("await runIllustrationThinkingCheck('llm3');")
                        assert len(checks) == 6 and checks[-1]["target"] == "llm3"
                        assert await evaluate("return configReads===2;")
                        assert not requests and not exceptions, (requests, exceptions)
                        print("[THINK_CHECK_QA] PASS: placement, five evidence/failure cases, deduplication, sequential all-check, safe model text, stale-result reset, per-slot saved config reload; only mocked provider requests")
            finally:
                process.terminate()
                process.wait(timeout=15)
                print(f"[ILLUST_PREVIEW_QA] Chromium stopped: pid={process.pid}, exit={process.returncode}")
    finally:
        await runner.cleanup()
        print("[ILLUST_PREVIEW_QA] Local QA server stopped")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except BaseException as exc:
        print(f"[ILLUST_PREVIEW_QA] Failed: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise
