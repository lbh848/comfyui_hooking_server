"""Local visual QA harness; no production server, settings, LLM, or GPU calls."""
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

from aiohttp import web

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import illustration_flow as flow

clients = set()
release = asyncio.Event()
jobs = set()


async def notify(kind, data):
    for client in list(clients):
        await client.send_json({"type": kind, "data": data})


flow.configure(notify)


@flow.llm_call
async def call(name, messages, *, wait=False, fallback=False):
    flow.llm_attempt({"type": "attempt_start", "slot": "llm1", "phase": "primary", "attempt": 1}, "primary-model", "test-provider")
    flow.llm_metadata(status="processing")
    if wait:
        await release.wait()
    if fallback:
        flow.llm_attempt({"type": "attempt_failure", "slot": "llm1", "phase": "primary", "raw_response": "검증용 실패 응답", "reason": "응답 파싱 실패"}, "primary-model", "test-provider")
        flow.llm_attempt({"type": "attempt_start", "slot": "llm2", "phase": "fallback", "attempt": 1}, "fallback-model", "test-provider-2")
        flow.llm_metadata(status="processing")
    return "장면 처리 결과입니다. <script>텍스트로 표시</script>"


class Runner:
    @flow.queue_execution
    async def run(self, job):
        await call("CHARACTER-RESOLVE", [{"role": "user", "content": "검증용 캐릭터 문맥"}])
        gate = asyncio.Semaphore(2)
        async def worker(index):
            async with gate:
                return await call(f"CALL2-DETAIL {index}", [{"role": "user", "content": f"장면 {index}의 원문 입력"}], wait=index > 1, fallback=index == 1)
        await flow.gather(*(flow.create_task(worker(i), flow_label=f"CALL2-DETAIL {i}") for i in range(1, 5)))
        await call("CALL3 · 결과 병합", [])
        return {"success": True, "count": 4}


async def page(request):
    return web.Response(content_type="text/html", text='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>삽화 흐름 UI 검증</title>
<style>body{background:#0b1220;color:#e2e8f0;font:14px system-ui;padding:28px}button{padding:10px;margin:5px;cursor:pointer}#toast{position:fixed;bottom:25px;right:25px;background:#713f12;color:white;padding:15px;z-index:99999}</style>
<h1>삽화 흐름 UI 검증</h1><p>실제 그래프 추적 코드와 UI를 사용한 로컬 테스트입니다.</p>
<button onclick="fetch('/start',{method:'POST'})">새 요청 시작</button><button onclick="openIllustrationFlow()">삽화 흐름 보기</button><button onclick="fetch('/finish',{method:'POST'})">대기 작업 완료</button><button onclick="showToast('삽화를 생성하려면 먼저 활성 봇을 선택해야 합니다.');openIllustrationFlow()">봇 미선택 경고 검증</button>
<script src="/illustration_flow.js"></script><script>
function showToast(message){let t=document.getElementById('toast');if(!t){t=document.createElement('div');t.id='toast';document.body.append(t)}rehomeIllustrationToast();t.textContent=message}
const ws=new WebSocket('ws://'+location.host+'/ws');ws.onmessage=e=>{const m=JSON.parse(e.data);receiveIllustrationFlow(m.data,true)};
</script></html>''')


async def script(request):
    return web.FileResponse(ROOT / "frontend" / "illustration_flow.js")


async def state(request):
    if request.query.get("node"):
        node = flow.detail(request.query.get("run"), request.query["node"])
        return web.json_response({"node": node}, status=200 if node else 404)
    return web.json_response({"flow": flow.snapshot()})


async def start(request):
    release.clear()
    job = SimpleNamespace(id=str(len(jobs)) + str(asyncio.get_running_loop().time()), label="삽화 요청 · 병렬 4개", type="illustration_llm_build", params={}, status="pending")
    flow.queue_added(job)
    task = asyncio.create_task(Runner().run(job)); jobs.add(task); task.add_done_callback(jobs.discard)
    return web.json_response({"ok": True})


async def finish(request):
    release.set()
    return web.json_response({"ok": True})


async def ws(request):
    socket = web.WebSocketResponse(); await socket.prepare(request); clients.add(socket)
    try:
        async for _ in socket:
            pass
    finally:
        clients.discard(socket)
    return socket


app = web.Application()
app.add_routes([web.get('/', page), web.get('/illustration_flow.js', script), web.get('/api/illustration_flow', state), web.get('/ws', ws), web.post('/start', start), web.post('/finish', finish)])
if __name__ == '__main__':
    web.run_app(app, host='127.0.0.1', port=8196)
