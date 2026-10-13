"""Browser QA with the actual preview endpoint and saved data, no real work."""
import ast
import asyncio
import base64
import copy
from contextlib import redirect_stdout
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import traceback
from aiohttp import ClientSession, web

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from modes import illustration_context_pipeline, illustration_original_assets
from queue_manager import queue_manager
import workflow_profiles


async def main():
    saved = json.loads((ROOT / "config.json").read_text(encoding="utf-8-sig"))
    config = copy.deepcopy(saved)
    bot_module = importlib.import_module("modes.bot_mode")
    def read_bots():
        return json.loads(Path(bot_module.BOT_DATA_FILE).read_text(encoding="utf-8"))
    # Execute the production handlers in isolation; do not import/start server.py.
    tree = ast.parse((ROOT / "server.py").read_text(encoding="utf-8"))
    functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and node.name in {"_capture_illustration_runtime_snapshot", "handle_api_illustration_execution_preview"}]
    namespace = dict(web=web, copy=copy, traceback=traceback, os=os, app_config=config,
        workflow_profiles=workflow_profiles, illustration_context_pipeline=illustration_context_pipeline,
        illustration_original_assets=illustration_original_assets, queue_manager=queue_manager,
        bot_mode=None, BASE_DIR=str(ROOT), _illustration_quality_inspection_runtime_enabled=False,
        _load_word_rules_snapshot=lambda name: [], _load_bot_data_readonly=read_bots)
    exec(compile(ast.Module(body=functions, type_ignores=[]), "isolated-preview-handlers", "exec"), namespace)
    html = (ROOT / "frontend/index.html").read_text(encoding="utf-8")
    css = re.search(r"<style>(.*?)</style>", html, re.S).group(1)
    start = html.index("        async function renderIllustrationNodePreview()")
    render = html[start:html.index("\n        }", start) + len("\n        }")]
    start = html.index("        function llmRouteResolvedFallback(")
    fallback = html[start:html.index("\n        }", start) + len("\n        }")]
    start = html.index('                <section class="illust-flow-preview" aria-labelledby=')
    section = html[start:html.index('</section>`;', start) + len('</section>')]
    public = {key: value for key, value in saved.items() if key == "llm_routing" or re.fullmatch(r"llm_(max_concurrency|model)\d*", key)}
    page = "<html lang='ko'><meta charset='utf-8'><style>" + css + "</style><body style='background:var(--bg);color:var(--text);padding:20px'><div class='modal visible' style='display:block;position:static'><div id='illust-tab-content-toggles' class='illust-settings'>" + section + "</div></div><script src='/illustration_preview.js'></script><script src='/llm_thinking_check.js'></script><script>"
    page += "const LLM_SLOTS=[1,2,3,4,5,6,7,8,9,10]; let _illustrationPreviewRequestId=0; let _illustrationSettingsActiveGroup='call2_detail'; let _illustrationContextToggles=" + json.dumps(saved.get("illustration_context_toggles") or {}) + "; let currentConfig=" + json.dumps(public) + ";\n" + fallback + "\n" + render + "\nrenderIllustrationNodePreview();</script></body></html>"
    requests, plans = [], []
    log_path = ROOT / f"요구사항/backup_actual_execution_preview_20261003/browser_output_{int(time.time())}.log"
    async def preview(request):
        with log_path.open("a", encoding="utf-8") as output, redirect_stdout(output):
            response = await namespace["handle_api_illustration_execution_preview"](request)
        assert response.status == 200, response.text
        plans.append(json.loads(response.text))
        return response
    async def forbidden(request):
        requests.append(request.path)
        print(f"[EXECUTION_PREVIEW_QA] Unexpected API: {request.path}")
        return web.json_response({"error": "unexpected API"}, status=500)
    app = web.Application()
    app.router.add_get("/", lambda request: web.Response(text=page, content_type="text/html"))
    app.router.add_get("/illustration_preview.js", lambda request: web.FileResponse(ROOT / "frontend/illustration_preview.js"))
    app.router.add_get("/llm_thinking_check.js", lambda request: web.FileResponse(ROOT / "frontend/llm_thinking_check.js"))
    app.router.add_post("/api/illustration_context/execution-preview", preview)
    app.router.add_route("*", "/api/{path:.*}", forbidden)
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
        with tempfile.TemporaryDirectory(prefix="actual-illust-preview-") as profile:
            process = subprocess.Popen([str(browsers[-1]), "--headless", "--disable-gpu", "--no-first-run",
                "--remote-debugging-port=0", f"--user-data-dir={profile}", "about:blank"], stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
            try:
                active_port = Path(profile) / "DevToolsActivePort"
                for _ in range(100):
                    if active_port.exists():
                        break
                    await asyncio.sleep(0.1)
                debug_port = active_port.read_text(encoding="utf-8").splitlines()[0]
                async with ClientSession() as session:
                    target = None
                    for _ in range(100):
                        async with session.get(f"http://127.0.0.1:{debug_port}/json/list") as response:
                            target = next((item for item in await response.json() if item["type"] == "page"), None)
                        if target:
                            break
                        await asyncio.sleep(0.1)
                    if not target:
                        raise RuntimeError("Chromium page target did not become ready")
                    async with session.ws_connect(target["webSocketDebuggerUrl"]) as ws:
                        sequence, exceptions = 0, []
                        async def cdp(method, params=None):
                            nonlocal sequence
                            sequence += 1
                            await ws.send_json({"id": sequence, "method": method, "params": params or {}})
                            while True:
                                result = await asyncio.wait_for(ws.receive_json(), 20)
                                if result.get("method") == "Runtime.exceptionThrown":
                                    exceptions.append(result["params"])
                                if result.get("id") == sequence:
                                    assert "error" not in result, result
                                    return result.get("result", {})
                        async def evaluate(code):
                            result = await cdp("Runtime.evaluate", {"expression": "(async()=>{" + code + "})()", "awaitPromise": True, "returnByValue": True})
                            assert "exceptionDetails" not in result, result
                            return result.get("result", {}).get("value")
                        await cdp("Runtime.enable")
                        await cdp("Emulation.setDeviceMetricsOverride", {"width": 1440, "height": 1600, "deviceScaleFactor": 1, "mobile": False})
                        await cdp("Page.navigate", {"url": f"http://127.0.0.1:{port}/"})
                        for _ in range(100):
                            if await evaluate("return !!document.querySelector('.illust-preview-state');"):
                                break
                            await asyncio.sleep(0.05)
                        assert plans and plans[0]["nodes"]
                        character_controls = await evaluate("""
                            const count=document.querySelector('[data-illust-preview-scenario="character_count"]');
                            return {value:count?.value,disabled:count?.disabled,
                                scene:!!document.querySelector('[data-illust-preview-scenario="scene_count"]'),
                                slots:!!document.querySelector('[data-illust-preview-scenario="slot_count"]')};
                        """)
                        assert character_controls == {"value": "2", "disabled": False, "scene": False, "slots": False}, character_controls
                        assert plans[0]["scenario"]["character_count"] <= 2
                        print(f"[EXECUTION_PREVIEW_QA] Saved configuration: {plans[0]['summary']}")
                        initial_profile = next(node for node in plans[0]["nodes"] if node["name"] == "PROFILE-RESOLVE")
                        profile_display = await evaluate("""
                            const card=document.querySelector('[data-illust-preview-node="PROFILE-RESOLVE"]');
                            return {activation:card.querySelector('[data-activation]').dataset.activation,
                                state:card.querySelector('.illust-preview-state').textContent};
                        """)
                        assert profile_display["activation"] == ("on" if initial_profile["enabled"] else "off"), profile_display
                        assert "조건 대기" not in profile_display["state"], profile_display
                        initial_edges = await evaluate("return [...document.querySelectorAll('.illust-preview-edge')].map(edge=>({from:edge.dataset.fromNode,to:edge.dataset.toNode,simulated:edge.dataset.simulated}));")
                        translation = next(node for node in plans[0]["nodes"] if node["name"] == "PROFILE-CONTEXT-TRANSLATE")
                        if translation["enabled"]:
                            assert any(edge["from"] == translation["id"] and edge["to"] == initial_profile["id"]
                                       and edge["simulated"] == "true" for edge in initial_edges), initial_edges
                        # Change only in-memory test settings; production files remain read-only.
                        config["llm_max_concurrency3"] = 3
                        config["llm_model3"] = "QA model"
                        config["llm_routing"] = {task: {"primary": "llm3"} for task in illustration_context_pipeline._CALL_TASK_KEYS.values()}
                        config["illustration_workflow_type"] = "v3"
                        config["postprocess_enabled"] = True
                        observed = await evaluate("""
                            currentConfig.llm_max_concurrency3=3;currentConfig.llm_model3='QA model';
                            currentConfig.llm_routing=Object.fromEntries(getIllustrationPreviewTasks(['CHARACTER-RESOLVE','PROFILE-CONTEXT-TRANSLATE','PROFILE-RESOLVE','CALL1','CALL2-SELECT','CALL2-PLAN','CALL2-DETAIL','CALL2-KEYVIS','CALL3','MULTI-CHAR-MASK','ORIGINAL-ASSET']).map(task=>[task,{primary:'llm3'}]));
                            _illustrationContextToggles={..._illustrationContextToggles,illustration_output_mode:'both',output_count_min:11,output_count_max:11,
                                call2_parallel_enabled:true,call2_parallel_max_concurrency:3,call2_detail_parallel_enabled:true,
                                call2_detail_parallel_max_concurrency:5,key_visual:true,speak_enabled:true};
                            await renderIllustrationNodePreview();
                            const timeline=document.querySelector('[data-illust-preview-progress]');
                            let peak=0;
                            for(let t=0;t<Number(timeline.max);t+=0.1){timeline.value=t;timeline.dispatchEvent(new Event('input'));
                                peak=Math.max(peak,document.querySelectorAll('[data-simulation-state="processing"] .illust-preview-llm').length);}
                            const result={peak,details:document.querySelectorAll('[data-illust-preview-node="CALL2-DETAIL"]').length,
                                keyvis:document.querySelectorAll('[data-illust-preview-node="CALL2-KEYVIS"]').length,
                                summary:document.querySelector('[data-illust-preview-summary]').textContent};
                            timeline.value=0;timeline.dispatchEvent(new Event('input'));return result;
                        """)
                        assert observed["peak"] <= 3 and observed["details"] == 5 and observed["keyvis"] == 1, observed
                        edges = await evaluate("return [...document.querySelectorAll('.illust-preview-edge')].map(edge=>[edge.dataset.fromNode,edge.dataset.toNode]);")
                        def connects(source, destination):
                            return any(left.startswith(source + ":") and right.startswith(destination + ":") for left, right in edges)
                        assert connects("CALL2-DETAIL", "CALL3"), edges
                        assert connects("CALL3", "대사 이미지 합성"), edges
                        assert connects("이미지 렌더링", "대사 이미지 합성"), edges
                        assert connects("대사 이미지 합성", "이미지 저장"), edges
                        assert not connects("이미지 렌더링", "CALL3") and not connects("SLOT", "CALL3"), edges
                        # Verify active-bot switching through the same real endpoint.
                        # These fixtures and changes exist only in memory.
                        selected = config.get("bot_selected")
                        namespace["_load_bot_data_readonly"] = lambda: {"bots": [
                            {"name": "QA single", "postprocess_vn": {"enabled": False},
                             "characters": [{"name": "Aster", "visual_cards": [{"id": "base"}]}]},
                            {"name": "QA multi", "postprocess_vn": {"enabled": True}, "characters": [{"name": "Birch",
                             "visual_cards": [{"id": "base", "selection_guide": "Initial costume."},
                                              {"id": "changed", "selection_guide": "After changing clothes."}]},
                                {"name": "Cedar", "visual_cards": [{"id": "only"}]},
                                {"name": "Dahlia", "visual_cards": [
                                    {"id": "base", "selection_guide": "Initial uniform."},
                                    {"id": "changed", "selection_guide": "After a change."}]}]},
                        ]}
                        for bot, activation in [("QA single", "off"), ("QA multi", "on"), ("QA single", "off")]:
                            config["bot_selected"] = bot
                            display = await evaluate("""
                                await renderIllustrationNodePreview();
                                const card=document.querySelector('[data-illust-preview-node="PROFILE-RESOLVE"]');
                                return {activation:card.querySelector('[data-activation]').dataset.activation,
                                    state:card.querySelector('.illust-preview-state').textContent,
                                    composition:document.querySelector('[data-illust-preview-node="대사 이미지 합성"] [data-activation]').dataset.activation,
                                    translationLink:!!document.querySelector('.illust-preview-edge[data-from-node="PROFILE-CONTEXT-TRANSLATE:1"][data-to-node="PROFILE-RESOLVE:1"][data-simulated="true"]')};
                            """)
                            assert display["activation"] == activation and "조건 대기" not in display["state"], display
                            assert display["composition"] == activation, display
                            if activation == "on":
                                assert display["state"].startswith("대기"), display
                                assert display["translationLink"], display
                                for count, selected_names, profile_jobs in [
                                    (0, [], 0), (1, ["Birch"], 1),
                                    (3, ["Birch", "Cedar", "Dahlia"], 2),
                                    (8, ["Birch", "Cedar", "Dahlia"], 2),
                                    (2, ["Birch", "Cedar"], 1)]:
                                    count_display = await evaluate(f"""
                                        const count=document.querySelector('[data-illust-preview-scenario="character_count"]');
                                        count.value='{count}';await count.onchange();
                                        return {{activation:document.querySelector('[data-illust-preview-node="PROFILE-RESOLVE"] [data-activation]').dataset.activation}};
                                    """)
                                    assert plans[-1]["scenario"]["current_characters"] == selected_names, plans[-1]["scenario"]
                                    assert plans[-1]["scenario"]["character_count"] == len(selected_names), plans[-1]["scenario"]
                                    assert len([job for job in plans[-1]["jobs"] if job["name"] == "PROFILE-RESOLVE"]) == profile_jobs
                                    assert count_display["activation"] == ("on" if profile_jobs else "off"), count_display
                                    assert plans[-1]["scenario"]["scene_count"] == 11
                                profile_flow = await evaluate("""
                                    const names=['CHARACTER-RESOLVE','PROFILE-CONTEXT-CACHE','PROFILE-CONTEXT-TRANSLATE','PROFILE-RESOLVE','CALL1'];
                                    const firstProcessing={}, lastProcessing={};
                                    const packets=new Set();
                                    const timeline=document.querySelector('[data-illust-preview-progress]');
                                    for(let time=0;time<Number(timeline.max);time+=0.02){
                                        timeline.value=time;timeline.dispatchEvent(new Event('input'));
                                        for(const name of names){
                                            const card=document.querySelector(`[data-illust-preview-node="${name}"]`);
                                            if(card?.dataset.simulationState==='processing'){
                                                if(firstProcessing[name]===undefined)firstProcessing[name]=time;
                                                lastProcessing[name]=time;
                                            }
                                        }
                                        for(const edge of document.querySelectorAll('.illust-preview-edge')){
                                            const packet=edge.nextElementSibling;
                                            if(packet?.classList.contains('illust-preview-packet') && getComputedStyle(packet).display!=='none')
                                                packets.add(`${edge.dataset.fromNode}->${edge.dataset.toNode}`);
                                        }
                                    }
                                    timeline.value=0;timeline.dispatchEvent(new Event('input'));
                                    return {steps:names.map(name=>({name,time:firstProcessing[name],
                                        duration:lastProcessing[name]-firstProcessing[name]+0.02})),packets:[...packets]};
                                """)
                                steps = profile_flow["steps"]
                                assert all("time" in step for step in steps), profile_flow
                                assert all(left["time"] < right["time"] for left, right in zip(steps, steps[1:])), profile_flow
                                cache_duration = next(step["duration"] for step in steps if step["name"] == "PROFILE-CONTEXT-CACHE")
                                assert all(cache_duration * 10 < step["duration"] for step in steps
                                           if step["name"] != "PROFILE-CONTEXT-CACHE"), profile_flow
                                for source, target in zip(steps, steps[1:]):
                                    assert f"{source['name']}:1->{target['name']}:1" in profile_flow["packets"], profile_flow
                        namespace["_load_bot_data_readonly"] = read_bots
                        config["bot_selected"] = selected
                        dialogue_off = await evaluate("""
                            _illustrationContextToggles.speak_enabled=false;await renderIllustrationNodePreview();
                            return {activation:document.querySelector('[data-illust-preview-node="CALL3"] [data-activation]').dataset.activation,
                                composition:document.querySelector('[data-illust-preview-node="대사 이미지 합성"] [data-activation]').dataset.activation,
                                rendering:document.querySelector('[data-illust-preview-node="이미지 렌더링"] [data-activation]').dataset.activation,
                                edges:[...document.querySelectorAll('.illust-preview-edge')].map(edge=>[edge.dataset.fromNode,edge.dataset.toNode])};
                        """)
                        assert dialogue_off["activation"] == "off", dialogue_off
                        assert dialogue_off["composition"] == "off" and dialogue_off["rendering"] == "on", dialogue_off
                        assert not any(part.startswith("CALL3:") for edge in dialogue_off["edges"] for part in edge), dialogue_off
                        await evaluate("_illustrationContextToggles.speak_enabled=true;")
                        config["postprocess_enabled"] = False
                        master_off = await evaluate("""
                            await renderIllustrationNodePreview();
                            return {call3:document.querySelector('[data-illust-preview-node="CALL3"] [data-activation]').dataset.activation,
                                composition:document.querySelector('[data-illust-preview-node="대사 이미지 합성"] [data-activation]').dataset.activation};
                        """)
                        assert master_off == {"call3": "on", "composition": "off"}, master_off
                        config["postprocess_enabled"] = True
                        for mode in ["original_asset", "illustration", "both"]:
                            result = await evaluate(f"_illustrationContextToggles.illustration_output_mode='{mode}';await renderIllustrationNodePreview();return {{mode:'{mode}',image:document.querySelector('[data-illust-preview-node=\"이미지 렌더링\"]').dataset.enabled,asset:document.querySelector('[data-illust-preview-node=\"ORIGINAL-ASSET\"]').dataset.enabled}};")
                            assert (result["image"] == "true") == (mode != "original_asset"), result
                        for protocol, disabled in [("prompt_batch_v1", True), ("context", False)]:
                            protocol_display = await evaluate(f"""
                                const protocol=document.querySelector('[data-illust-preview-scenario="protocol"]');
                                protocol.value='{protocol}';await protocol.onchange();
                                return {{disabled:document.querySelector('[data-illust-preview-scenario="character_count"]').disabled,
                                    character:document.querySelector('[data-illust-preview-node="CHARACTER-RESOLVE"] [data-activation]').dataset.activation}};
                            """)
                            assert protocol_display["disabled"] == disabled, protocol_display
                            assert protocol_display["character"] == ("off" if disabled else "on"), protocol_display
                        await evaluate("document.querySelector('[data-illust-preview-scenario=conditional]').checked=true;await renderIllustrationNodePreview();document.querySelector('[data-illust-preview-play]').click();")
                        await asyncio.sleep(0.3)
                        assert await evaluate("return document.getElementById('illust-node-preview').dataset.simulationPlaying==='true';")
                        await evaluate("document.querySelector('[data-illust-preview-reset]').click();")
                        await evaluate("document.querySelector('[data-illust-preview-scenario=conditional]').checked=false;await renderIllustrationNodePreview();")
                        state_colors = await evaluate("""
                            const card=document.querySelector('[data-illust-preview-node="이미지 렌더링"]');
                            const waitState=card.dataset.simulationState, waitColor=getComputedStyle(card).borderLeftColor;
                            const timeline=document.querySelector('[data-illust-preview-progress]');
                            let activeColor;
                            for(let time=0;time<Number(timeline.max);time+=0.1){timeline.value=time;timeline.dispatchEvent(new Event('input'));
                                if(card.dataset.simulationState==='processing'){activeColor=getComputedStyle(card).borderLeftColor;break;}}
                            const result={waitState,waitColor,activeColor};
                            timeline.value=0;timeline.dispatchEvent(new Event('input'));return result;
                        """)
                        assert state_colors["waitState"] == "waiting" and state_colors["activeColor"] != state_colors["waitColor"], state_colors
                        assert await evaluate("return !!document.querySelector('[data-illust-thinking-checks]') && !document.querySelector('.illust-thinking-node-badge');")
                        screenshot = await cdp("Page.captureScreenshot", {"format": "png"})
                        artifact = ROOT / f"요구사항/backup_actual_execution_preview_20261003/browser_{int(time.time())}.png"
                        artifact.write_bytes(base64.b64decode(screenshot["data"]))
                        assert not requests and not exceptions, (requests, exceptions)
                        assert config is namespace["app_config"]
                        print(f"[EXECUTION_PREVIEW_QA] PASS: default character count 2, count changes update CURRENT/profile work, no scene/slot controls, saved settings, profile animation order, bot switching, composition settings, shared cap 3, all output modes, playback, screenshot {artifact}")
            finally:
                process.terminate()
                process.wait(timeout=15)
                print(f"[EXECUTION_PREVIEW_QA] Chromium stopped: pid={process.pid}, exit={process.returncode}")
    finally:
        await runner.cleanup()
        print("[EXECUTION_PREVIEW_QA] QA server stopped")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except BaseException as e:
        print(f"[EXECUTION_PREVIEW_QA] Failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        raise
