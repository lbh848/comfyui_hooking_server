"""Exercise the real flow UI in local headless Chromium, with no LLM/GPU calls."""
import asyncio
import base64
import copy
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
    if name == "CALL1":
        started = time.time() - 25
        flow.llm_metadata(started_at=started, thinking_started_at=started + 2,
            thinking_ended_at=started + 5, response_started_at=started + 10,
            response_ended_at=started + 17)
    return f"{name} 완료 · <script>텍스트로 표시</script>"


class Runner:
    @flow.queue_execution
    async def execute(self, item):
        return await item.handler()


def illustration_layout_fixture():
    columns = [
        ["CHARACTER-RESOLVE", "PROFILE-CONTEXT-CACHE", "PROFILE-RESOLVE", "PROFILE-RESOLVE"],
        ["CALL1", "CALL2-SELECT"],
        ["CALL2-PLAN 1/3", "CALL2-PLAN 2/3", "CALL2-PLAN 3/3"],
        ["CALL2-DETAIL", "CALL2-KEYVIS", "CALL3"],
        ["MULTI-CHAR-MASK slot=-1", "MULTI-CHAR-MASK slot=7", "ORIGINAL-ASSET", "ORIGINAL-ASSET-RECOVERY"],
        ["삽화 1/2 · slot -1", "삽화 2/2 · slot 7"],
    ]
    nodes = [{"id": "fixture-root", "label": "삽화 요청", "kind": "request", "dependencies": [], "status": "completed"}]
    expected = {}
    previous = [nodes[0]["id"]]
    for step, labels in enumerate(columns, 1):
        current = []
        for index, label in enumerate(labels):
            node_id = f"step-{step}-{index}"
            node = {
                "id": node_id, "label": label, "kind": "llm", "executor": "llm",
                "model": "QA model", "status": "completed", "dependencies": previous[:],
            }
            if step == 6:
                node.update(kind="image", executor="comfy", model="ComfyUI")
                node["layout_group"] = "illustration_images"
                node["prompt_id"] = f"fixture-prompt-{index}"
            if label == "CALL3":
                node["layout_order"] = -100  # Existing metadata must not reorder STEP 5.
            nodes.append(node)
            current.append(node_id)
            expected[node_id] = step
        previous = current
    images = previous[:]
    posts = []
    for index in (1, 0):
        node_id = f"post-{index}"
        nodes.append({
            "id": node_id, "label": "이미지 후처리", "kind": "stage", "executor": "process",
            "status": "completed", "dependencies": images + posts,
            "prompt_id": f"fixture-prompt-{index}",
        })
        posts.append(node_id)
        expected[node_id] = 7
    nodes.append({"id": "fixture-result", "label": "결과 반환", "kind": "result", "status": "completed", "dependencies": posts})
    expected["fixture-result"] = 8
    return {
        "id": "six-step-layout-fixture", "kind": "illustration", "label": "삽화 요청 · 단계별 배치 검증",
        "status": "completed", "revision": 1, "created_at": time.time() + 1000,
        "nodes": nodes,
    }, expected


async def prepare_graphs():
    illustration = job("삽화 요청", "illustration_llm_build")

    @flow.stage("이미지 생성", executor="comfy", layout_group="illustration_generation")
    async def generate(positive, negative, provider="comfy"):
        return b"qa image bytes", ""

    @flow.stage("이미지 후처리")
    async def deliver():
        return {"delivered": True}

    async def illustrate():
        await call("CALL1", [{"role": "user", "content": "삽화 원문"}])
        from modes.illustration_scene_pipeline import fill_state_table
        flow.record_call1_state_table(fill_state_table({
            "SLOT0": {"서연": {"outfit": "blue coat, white shirt", "hair": "brown hair, braid"},
                      "Jun": {"outfit": "brown jacket", "hair": "short hair"}},
            "SLOT2": {"서연": {"outfit": "white shirt"}},
            "SLOT3": {"서연": {"hair": "brown hair, hair down"}},
            "SLOT5": {"서연": {"outfit": ""}},
        }, ["서연", "Jun"], list(range(8))))
        images = []
        for slot in (-1, 7):
            image = job(f"삽화 {len(images) + 1}/2 · slot {slot}", "illustration", {
                "provider": "comfy", "prompt_data": {"slot": slot},
            })
            async def render_image(slot=slot):
                await generate(f"QA scene prompt {slot}", "QA negative prompt")
                return {"success": True, "prompt_id": f"qa-image-{slot}"}
            image.handler = render_image
            images.append(image)
        await asyncio.gather(*(Runner().execute(image) for image in images))
        flow.merge(images)
        await deliver()
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
    reader = {"delay": 0, "fail": False, "active": {}, "maximum": {}, "requests": {}, "version": 0}
    live_originals = {}
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
            node_id = request.query["node"]
            reader["active"][node_id] = reader["active"].get(node_id, 0) + 1
            reader["maximum"][node_id] = max(reader["maximum"].get(node_id, 0), reader["active"][node_id])
            reader["requests"][node_id] = reader["requests"].get(node_id, 0) + 1
            try:
                await asyncio.sleep(reader["delay"])
                if reader["fail"]:
                    reader["fail"] = False
                    print("[FLOW_READER_QA] Injected transient detail HTTP 500")
                    return web.json_response({"error": "QA transient failure"}, status=500)
                node = flow.detail(request.query.get("run"), node_id)
                return web.json_response({"node": node}, status=200 if node else 404)
            finally:
                reader["active"][node_id] -= 1
        return web.json_response({"flow": flow.snapshot(kind=request.query.get("kind", "illustration"))})

    async def reader_update(request):
        run = flow._latest
        call1 = next(node for node in run["nodes"].values() if node["label"] == "CALL1")
        images = [node for node in run["nodes"].values() if node["kind"] == "image"]
        mode = request.query.get("mode")
        reader["version"] += 1
        if mode == "prepare":
            run["status"] = "processing"
            reader["delay"] = 0.12
            flow.update(run, call1["id"], input="\n".join(f"Reader input line {i}" for i in range(200)),
                output="\n".join(f"Reader output line {i}" for i in range(200)),
                attempts=[{"type": "attempt_start", "model": "QA model", "phase": "primary",
                           "diagnostics": [f"Reader diagnostic line {i}" for i in range(200)]}])
        elif mode == "related":
            flow.update(run, call1["id"], output=call1["output"] + "\nUPDATED READER OUTPUT",
                attempts=call1["attempts"] + [{"type": "attempt_success", "phase": "primary", "model": "QA model"}])
        elif mode == "error":
            reader["fail"] = True
            flow.update(run, images[0]["id"], progress=reader["version"])
        elif mode == "restore":
            reader["delay"] = 0
            run["status"] = "completed"
            for node in images:
                flow.update(run, node["id"], status="completed", progress=100)
        else:
            flow.update(run, images[0]["id"], status="processing", ended_at=None, progress=reader["version"])
        return web.json_response({"flow": flow.snapshot(), "reader": reader, "call1_id": call1["id"]})

    async def reader_stats(request):
        return web.json_response(reader)

    async def live_update(request):
        kind = request.query.get("kind", "illustration")
        run = {"illustration": flow._latest, "video": flow._latest_video,
               "video_input": flow._latest_video_input}[kind]
        mode = request.query.get("mode", "stream")
        node = next(n for n in run["nodes"].values() if n.get("kind") == "llm")
        if mode == "prepare":
            live_originals[kind] = (copy.deepcopy(run["nodes"]), run["status"])
            run["status"] = "processing"
            now = time.time()
            flow.update(run, node["id"], status="processing", ended_at=None,
                        response_started_at=now, response_ended_at=now, model="Live model 0")
        elif mode == "add":
            flow.add_node(run, "병렬 검사", node_id="live-extra", kind="llm", executor="llm",
                          dependencies=[node["id"]], layout_group="live_extra")
        elif mode == "remove":
            del run["nodes"]["live-extra"]
            flow.changed(run)
        elif mode == "restore":
            run["nodes"], run["status"] = live_originals.pop(kind)
            flow.changed(run)
        else:
            index = int(request.query.get("index", "1"))
            flow.update(run, node["id"], response_ended_at=node["response_started_at"] + index,
                        model=f"Live model {index}", status="completed" if index == 20 else "processing")
        return web.json_response({"flow": flow.snapshot(kind=kind), "node_id": node["id"]})

    app = web.Application()
    app.add_routes([web.get("/", page), web.get("/illustration_flow.js", script), web.get("/api/illustration_flow", state),
                    web.post("/qa/reader", reader_update), web.get("/qa/reader/stats", reader_stats),
                    web.post("/qa/live", live_update)])
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
                    target = None
                    for attempt in range(100):
                        async with session.get(f"http://127.0.0.1:{debug_port}/json/list") as response:
                            targets = await response.json()
                        target = next((t for t in targets if t["type"] == "page"), None)
                        if target:
                            break
                        if attempt == 0:
                            print(f"[FLOW_QA] Waiting for Chromium page: targets={targets}")
                        await asyncio.sleep(0.05)
                    if target is None:
                        raise RuntimeError(f"Chromium page unavailable: targets={targets}")
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
                        for live_kind in ("illustration", "video", "video_input"):
                            await evaluate(f"document.querySelector('#if-tab-{live_kind}').click();window.liveKind={json.dumps(live_kind)};")
                            await evaluate("""
                                window.liveUpdate=async(mode,index=0)=>{
                                    const r=await fetch(`/qa/live?kind=${liveKind}&mode=${mode}&index=${index}`,{method:'POST'});
                                    const value=await r.json();receiveIllustrationFlow(value.flow,false,liveKind);return value;
                                };
                                const value=await liveUpdate('prepare');
                                window.liveCard=[...document.querySelectorAll('.if-node')].find(n=>n.dataset.nodeId===value.node_id);
                                window.livePort=liveCard.querySelector('.if-port');
                                window.liveCanvas=document.querySelector('.if-canvas');
                                window.liveSpace=document.querySelector('.if-space');
                                window.liveEdge=document.querySelector('.if-edges path');
                                livePort.focus({preventScroll:true});livePort.scrollIntoView({block:'nearest',inline:'nearest'});
                                window.liveTooltip=document.querySelector('.if-tooltip');
                                window.liveScroll=[document.querySelector('.if-viewport').scrollLeft,document.querySelector('.if-viewport').scrollTop];
                                window.liveDetachments=0;
                                window.liveObserver=new MutationObserver(records=>{
                                    for(const record of records)for(const node of record.removedNodes)
                                        if(node===liveCard||node.contains?.(liveCard)||node===livePort||node.contains?.(livePort))liveDetachments++;
                                });
                                liveObserver.observe(document.querySelector('.if-viewport'),{childList:true,subtree:true});
                            """)
                            live_bounds = await evaluate("const r=livePort.getBoundingClientRect();return {x:r.left+r.width/2,y:r.top+r.height/2};")
                            await cdp("Input.dispatchMouseEvent", {"type": "mouseMoved", **live_bounds})
                            await cdp("Input.dispatchMouseEvent", {"type": "mousePressed", "button": "left", "clickCount": 1, **live_bounds})
                            await evaluate("window.liveTooltip=document.querySelector('.if-tooltip');")
                            await evaluate("for(let i=1;i<=20;i++){await liveUpdate('stream',i);await new Promise(r=>setTimeout(r,10));}")
                            stable = await evaluate("""
                                const viewport=document.querySelector('.if-viewport');
                                return {connected:liveCard.isConnected&&livePort.isConnected,
                                    sameCanvas:document.querySelector('.if-canvas')===liveCanvas,
                                    sameSpace:document.querySelector('.if-space')===liveSpace,
                                    sameEdge:document.querySelector('.if-edges path')===liveEdge,
                                    detachments:liveDetachments,focus:document.activeElement===livePort,
                                    sameTooltip:document.querySelector('.if-tooltip')===liveTooltip,
                                    latestTooltip:document.querySelector('.if-tooltip')?.textContent.includes('Live model 20'),
                                    latestState:liveCard.dataset.status==='completed'&&liveCard.querySelector('.if-node-state').textContent.includes('완료'),
                                    latestTiming:liveCard.querySelector('.if-node-timing').textContent.includes('응답 20.0s'),
                                    latestModel:liveCard.querySelector('.if-node-model').textContent==='Live model 20',
                                    scroll:viewport.scrollLeft===liveScroll[0]&&viewport.scrollTop===liveScroll[1]};
                            """)
                            assert all(value for key, value in stable.items() if key != "detachments") and stable["detachments"] == 0, (live_kind, stable)
                            await cdp("Input.dispatchMouseEvent", {"type": "mouseReleased", "button": "left", "clickCount": 1, **live_bounds})
                            for _ in range(100):
                                if await evaluate("return document.querySelector('.if-detail').open&&document.querySelector('.if-detail h2').textContent===liveCard.querySelector('.if-node-title').textContent;"):
                                    break
                                await asyncio.sleep(0.02)
                            assert await evaluate("return document.querySelector('.if-detail').open&&document.querySelector('.if-detail-body').textContent.includes('Live model 20');"), live_kind
                            await evaluate("document.querySelector('.if-detail').close();livePort.focus({preventScroll:true});")
                            await evaluate("await liveUpdate('add');await liveUpdate('remove');")
                            assert await evaluate("return liveCard.isConnected&&livePort.isConnected&&liveDetachments===0&&document.activeElement===livePort&&!document.querySelector('.if-node[data-node-id=live-extra]');"), live_kind
                            await evaluate("""
                                liveObserver.disconnect();await liveUpdate('restore');
                                const r=await fetch(`/api/illustration_flow?kind=${liveKind}`);window.liveOriginal=(await r.json()).flow;
                                receiveIllustrationFlow({...liveOriginal,id:liveOriginal.id+'-new-run',revision:1},false,liveKind);
                            """)
                            assert await evaluate("return !liveCard.isConnected&&!livePort.isConnected&&!liveTooltip.isConnected;"), live_kind
                            await evaluate("receiveIllustrationFlow(liveOriginal,false,liveKind);")
                            print(f"[FLOW_LIVE_NODE_QA] {live_kind}: 20 streaming updates, real mouse click, stable nodes/ports/edges/hover/focus/scroll, add/remove and new-run reset PASS")
                        await evaluate("document.querySelector('#if-tab-illustration').click();")
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
                        assert await evaluate("return document.querySelectorAll('.if-node').length;") == 6
                        assert await evaluate("return [...document.querySelectorAll('.if-node-title')].filter(n=>n.textContent.startsWith('삽화 ')).map(n=>n.textContent);") == ["삽화 요청", "삽화 1/2 · slot -1", "삽화 2/2 · slot 7"]
                        assert await evaluate("return [...document.querySelectorAll('.if-node-title')].every(n=>n.textContent!=='이미지 생성');")
                        assert await evaluate("return [...document.querySelectorAll('.if-node-title')].some(n=>n.textContent==='이미지 후처리');")
                        assert await evaluate("return document.querySelector('.if-node-timing').textContent;") == "THINK 3.0s · 응답 7.0s"
                        bounds = await evaluate("""
                            const port=document.querySelector('.if-node-timing').closest('.if-node').querySelector('.if-port');
                            const r=port.getBoundingClientRect();return {x:r.left+r.width/2,y:r.top+r.height/2};
                        """)
                        await cdp("Input.dispatchMouseEvent", {"type": "mouseMoved", **bounds})
                        assert await evaluate("return document.querySelector('.if-tooltip').textContent.includes('추론 첫 수신:')&&document.querySelector('.if-tooltip').textContent.includes('본문 마지막 수신:');")
                        await evaluate("document.querySelector('.if-node-timing').closest('.if-node').querySelector('.if-port').click();")
                        for _ in range(100):
                            if await evaluate("return document.querySelector('.if-detail h2').textContent==='CALL1';"):
                                break
                            await asyncio.sleep(0.05)
                        assert await evaluate("""
                            const values=Object.fromEntries([...document.querySelectorAll('.if-meta dt')].map(n=>[n.textContent,n.nextElementSibling.textContent]));
                            return values['THINK 시간']==='3.0s'&&values['응답 작성 시간']==='7.0s'
                                &&/^[0-9]{2}:[0-9]{2}:[0-9]{2}[.][0-9]{3}$/.test(values['추론 첫 수신'])
                                &&/^[0-9]{2}:[0-9]{2}:[0-9]{2}[.][0-9]{3}$/.test(values['본문 마지막 수신']);
                        """)
                        timing_shot = await cdp("Page.captureScreenshot", {"format": "png"})
                        timing_artifact = ROOT / "tests/artifacts" / f"workflow_timing_qa_{time.time_ns()}.png"
                        timing_artifact.parent.mkdir(parents=True, exist_ok=True)
                        timing_artifact.write_bytes(base64.b64decode(timing_shot["data"]))
                        print(f"[FLOW_TIMING_QA] Card, hover and detail PASS: THINK 3.0s / response 7.0s; screenshot={timing_artifact}")
                        assert await evaluate("""
                            const table=document.querySelector('.if-state-table');
                            const rows=[...table.querySelectorAll('tbody tr')];
                            const cells=[...rows[0].querySelectorAll('td')];
                            return table.querySelectorAll('thead th').length===9 && rows.length===2
                                && rows[0].querySelector('th').textContent==='서연'
                                && cells[1].querySelector('[data-field=outfit]').dataset.source==='inherited'
                                && cells[2].querySelector('[data-field=outfit]').dataset.source==='direct'
                                && cells[2].querySelector('[data-field=hair]').dataset.source==='inherited'
                                && cells[3].querySelector('[data-field=hair]').dataset.source==='direct'
                                && cells[7].textContent.includes('(태그 없음)')
                                && rows[1].querySelectorAll('[data-source=direct]').length===2;
                        """)
                        await evaluate("document.querySelector('.if-detail-body').scrollTop=0;")
                        table_shot = await cdp("Page.captureScreenshot", {"format": "png"})
                        table_artifact = ROOT / "tests/artifacts" / f"call1_slot_table_qa_{time.time_ns()}.png"
                        table_artifact.write_bytes(base64.b64decode(table_shot["data"]))
                        print(f"[CALL1_TABLE_QA] Eight SLOTs, two people, per-field checkpoints, inheritance and clearing PASS; screenshot={table_artifact}")
                        await evaluate("document.querySelector('.if-detail').close(); document.querySelector('.if-call1-open').click();")
                        for _ in range(100):
                            if await evaluate("return !!document.querySelector('.if-state-table');"):
                                break
                            await asyncio.sleep(0.05)
                        assert await evaluate("return document.querySelector('.if-detail').open && !!document.querySelector('.if-state-table');")
                        await evaluate("""
                            const response=await fetch('/qa/reader?mode=prepare',{method:'POST'});
                            const value=await response.json();window.readerCall1=value.call1_id;
                            receiveIllustrationFlow(value.flow,false);
                        """)
                        for _ in range(100):
                            if await evaluate("return document.querySelector('.if-detail-body').textContent.includes('Reader output line 199');"):
                                break
                            await asyncio.sleep(0.02)
                        reader_loaded = await evaluate("return {text:document.querySelector('.if-detail-body').textContent.slice(-300),stats:await(await fetch('/qa/reader/stats')).json()};")
                        assert "Reader output line 199" in reader_loaded["text"], reader_loaded
                        await evaluate("""
                            window.readerNodes=()=>{
                                const body=document.querySelector('.if-detail-body');
                                const heading=[...body.querySelectorAll('h3')].find(n=>n.textContent==='입력');
                                return {body,table:body.querySelector('.if-state-table-wrap'),input:heading.nextElementSibling,
                                    output:[...body.children].filter(n=>n.tagName==='PRE').at(-1),
                                    trace:body.querySelector('details pre'),fold:body.querySelector('details')};
                            };
                            const n=readerNodes();n.fold.open=true;n.table.scrollLeft=430;
                            n.input.scrollTop=140;n.output.scrollTop=120;n.trace.scrollTop=90;n.body.scrollTop=200;
                            window.readerOriginal=n;
                            const target=n.table.querySelector('tbody td:nth-of-type(3) [data-field=outfit] div:last-child');
                            const selection=getSelection(),range=document.createRange();range.selectNodeContents(target);
                            selection.removeAllRanges();selection.addRange(range);window.readerSelection=selection.toString();
                        """)
                        await evaluate("""
                            for(let index=0;index<8;index++){
                                const r=await fetch('/qa/reader?mode=unrelated',{method:'POST'});
                                receiveIllustrationFlow((await r.json()).flow,false);
                                await new Promise(resolve=>setTimeout(resolve,20));
                            }
                            await new Promise(resolve=>setTimeout(resolve,450));
                        """)
                        assert await evaluate("""
                            const n=readerNodes(),old=readerOriginal;
                            const stats=await(await fetch('/qa/reader/stats')).json();
                            return n.table===old.table && n.input===old.input && n.output===old.output
                                && n.table.scrollLeft===430 && n.body.scrollTop===200
                                && n.input.scrollTop===140 && n.output.scrollTop===120 && n.trace.scrollTop===90
                                && n.fold.open && getSelection().toString()===readerSelection
                                && stats.maximum[readerCall1]===1 && stats.requests[readerCall1]>=2;
                        """)
                        await evaluate("""
                            const r=await fetch('/qa/reader?mode=related',{method:'POST'});
                            receiveIllustrationFlow((await r.json()).flow,false);
                            const n=readerNodes();n.table.scrollLeft=580;n.body.scrollTop=240;n.input.scrollTop=180;
                        """)
                        for _ in range(100):
                            if await evaluate("return readerNodes().output.textContent.includes('UPDATED READER OUTPUT');"):
                                break
                            await asyncio.sleep(0.02)
                        assert await evaluate("""
                            const n=readerNodes();return n.table!==readerOriginal.table && n.table.scrollLeft===580
                                && n.body.scrollTop===240 && n.input.scrollTop===180 && n.output.scrollTop===120
                                && n.trace.scrollTop===90 && n.fold.open && !n.body.querySelectorAll('details')[1].open;
                        """)
                        await evaluate("""
                            window.readerAfterUpdate=readerNodes();
                            const r=await fetch('/qa/reader?mode=error',{method:'POST'});
                            receiveIllustrationFlow((await r.json()).flow,false);
                            await new Promise(resolve=>setTimeout(resolve,180));
                        """)
                        assert await evaluate("return readerNodes().table===readerAfterUpdate.table && readerNodes().table.scrollLeft===580 && readerNodes().output.textContent.includes('UPDATED READER OUTPUT');")
                        print("[FLOW_READER_QA] Unchanged DOM/selection, CALL1 horizontal scroll, nested log scroll, folded state, request coalescing and transient error PASS")
                        await evaluate("""
                            const r=await fetch('/qa/reader?mode=unrelated',{method:'POST'});
                            receiveIllustrationFlow((await r.json()).flow,false);
                        """)
                        await evaluate("document.querySelector('.if-detail').close();")
                        await evaluate("document.querySelector('.if-port[data-node-id=\"삽화 1/2 · slot -1\"]').click();")
                        for _ in range(100):
                            if await evaluate("return document.querySelector('.if-detail h2').textContent==='삽화 1/2 · slot -1';"):
                                break
                            await asyncio.sleep(0.05)
                        assert await evaluate("""
                            const text=document.querySelector('.if-detail-body').textContent;
                            return text.includes('QA scene prompt -1') && text.includes('QA negative prompt')
                                && text.includes('generation') && text.includes('image_bytes')
                                && text.includes('qa-image--1') && text.includes('queue') && text.includes('처리 중');
                        """)
                        await evaluate("""
                            const r=await fetch('/qa/reader?mode=restore',{method:'POST'});
                            receiveIllustrationFlow((await r.json()).flow,false);
                        """)
                        await asyncio.sleep(0.18)
                        assert await evaluate("return document.querySelector('.if-detail h2').textContent==='삽화 1/2 · slot -1' && !document.querySelector('.if-state-table');")
                        print("[FLOW_READER_QA] Switching nodes while a previous detail request is pending PASS")
                        image_shot = await cdp("Page.captureScreenshot", {"format": "png"})
                        image_artifact = ROOT / "tests/artifacts" / f"workflow_merged_image_qa_{time.time_ns()}.png"
                        image_artifact.write_bytes(base64.b64decode(image_shot["data"]))
                        print(f"[FLOW_IMAGE_QA] One record per slot, prompts, generation and queue results PASS; screenshot={image_artifact}")
                        await evaluate("document.querySelector('.if-detail').close();")
                        await evaluate("document.querySelector('#if-tab-illustration').dispatchEvent(new KeyboardEvent('keydown', {key:'ArrowRight', bubbles:true}));")
                        assert await evaluate("return document.activeElement.id;") == "if-tab-video"
                        await evaluate("const buttons=[...document.querySelectorAll('.if-actions button')]; for(let i=0;i<3;i++) buttons.find(b=>b.textContent==='−').click();")
                        zoom = await evaluate("return document.querySelector('#if-zoom-reset').textContent;")
                        await evaluate("document.querySelector('#if-tab-illustration').click(); document.querySelector('#if-tab-video').click();")
                        assert await evaluate("return document.querySelector('#if-zoom-reset').textContent;") == zoom
                        await evaluate("const r=await fetch('/api/illustration_flow');const {flow}=await r.json();receiveIllustrationFlow({...flow,id:'new-illustration',created_at:flow.created_at+100,revision:1},true);")
                        assert await evaluate("return document.querySelector('#if-tab-video').getAttribute('aria-selected');") == "true"
                        fixture, expected = illustration_layout_fixture()
                        await evaluate(f"receiveIllustrationFlow({json.dumps(fixture, ensure_ascii=False)}, false);")
                        await evaluate("document.querySelector('#if-tab-illustration').click();")
                        assert await evaluate("return [...document.querySelectorAll('.if-node-timing')].every(n=>n.textContent==='THINK 0.0s · 응답 일괄');")
                        observed = await evaluate("return Object.fromEntries([...document.querySelectorAll('.if-node')].map(n=>[n.dataset.nodeId,{step:Number(n.dataset.step),x:parseFloat(n.style.left),y:parseFloat(n.style.top)}]));")
                        for node_id, step in expected.items():
                            assert observed[node_id]["step"] == step, (node_id, observed[node_id])
                        assert await evaluate("return [...document.querySelectorAll('.if-step-title')].filter(n=>n.textContent.startsWith('STEP ')).map(n=>n.textContent);") == [f"STEP {step}" for step in range(1, 7)]
                        assert await evaluate("return document.querySelector('.if-direction').textContent;") == "실행 순서 →"
                        first_y = {min(observed[node_id]["y"] for node_id, column in expected.items() if column == step) for step in range(1, 7)}
                        assert len(first_y) == 1, first_y
                        assert observed["step-5-0"]["y"] < observed["step-5-1"]["y"] < observed["step-5-2"]["y"]
                        for index in (0, 1):
                            assert observed[f"step-5-{index}"]["y"] == observed[f"step-6-{index}"]["y"] == observed[f"post-{index}"]["y"]
                        assert observed["post-0"]["x"] == observed["post-1"]["x"]
                        assert observed["fixture-result"]["y"] > observed["post-1"]["y"]
                        fixture["revision"] += 1
                        fixture["nodes"] = list(reversed(fixture["nodes"]))
                        await evaluate(f"receiveIllustrationFlow({json.dumps(fixture, ensure_ascii=False)}, false);")
                        reordered = await evaluate("return Object.fromEntries([...document.querySelectorAll('.if-node')].map(n=>[n.dataset.nodeId,{step:Number(n.dataset.step),x:parseFloat(n.style.left),y:parseFloat(n.style.top)}]));")
                        for node_id, step in expected.items():
                            assert reordered[node_id]["step"] == step
                            assert reordered[node_id] == observed[node_id]
                        await evaluate("const buttons=[...document.querySelectorAll('.if-actions button')]; for(let i=0;i<3;i++) buttons.find(b=>b.textContent==='−').click();")
                        screenshot = await cdp("Page.captureScreenshot", {"format": "png"})
                        artifact = ROOT / "요구사항" / f"workflow_flow_qa_{time.time_ns()}.png"
                        artifact.parent.mkdir(exist_ok=True)
                        artifact.write_bytes(base64.b64decode(screenshot["data"]))
                        await evaluate("""
                            document.querySelector('#if-zoom-reset').click();
                            document.querySelector('.if-viewport').scrollLeft=parseFloat(document.querySelector('.if-node[data-step="5"]').style.left)-24;
                            await new Promise(requestAnimationFrame); await new Promise(requestAnimationFrame);
                        """)
                        assert await evaluate("""
                            const viewport=document.querySelector('.if-viewport').getBoundingClientRect();
                            return [...document.querySelectorAll('.if-node[data-step="5"],.if-node[data-step="7"],.if-node[data-step="8"]')].every(n=>{
                                const r=n.getBoundingClientRect(); return r.left>=viewport.left && r.right<=viewport.right;
                            });
                        """)
                        lane_shot = await cdp("Page.captureScreenshot", {"format": "png"})
                        lane_artifact = ROOT / "tests/artifacts" / f"workflow_lane_alignment_qa_{time.time_ns()}.png"
                        lane_artifact.write_bytes(base64.b64decode(lane_shot["data"]))
                        print(f"[FLOW_LANE_QA] STEP5/STEP6/postprocess rows, shared work and shuffled input PASS; screenshot={lane_artifact}")
                        await cdp("Emulation.setDeviceMetricsOverride", {"width": 390, "height": 844, "deviceScaleFactor": 1, "mobile": False})
                        assert await evaluate("const r=document.querySelector('#illustration-flow-modal').getBoundingClientRect();return r.left>=0 && r.right<=innerWidth && r.bottom<=innerHeight;")
                        assert await evaluate("return document.querySelector('.if-footer').getBoundingClientRect().bottom <= document.querySelector('#illustration-flow-modal').getBoundingClientRect().bottom;")
                        await evaluate("document.querySelector('#if-tab-video').click();")
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
