"""Video graphs follow real queue handoffs and routed calls without generating media."""
import asyncio
import ast
import json
import traceback
from pathlib import Path
from types import SimpleNamespace

import pytest
from aiohttp import web

import illustration_flow as flow
from modes import llm_service
from queue_manager import QueueItem, QueueManager


@pytest.fixture(autouse=True)
def isolated_flow(monkeypatch):
    monkeypatch.setattr(flow, "_latest", None)
    monkeypatch.setattr(flow, "_latest_video", None)
    monkeypatch.setattr(flow, "_notify", None)
    monkeypatch.setattr(llm_service, "_current_config", {
        **llm_service._current_config,
        "llm_model": "primary-test", "llm_model2": "fallback-test",
        "llm_service": "test", "llm_service2": "fallback-provider",
        "llm_routing": {"video_prompt_i2v_compose": {
            "primary": "llm1", "fallback": "llm2", "max_retries": 0,
            "fallback_max_retries": 0, "retry_delay_sec": 0, "fallback_retry_delay_sec": 0,
        }},
    })


def job(name, kind="video_prompt_build", *, session_id=""):
    params = {"video_input_session_id": session_id} if session_id else {}
    item = QueueItem(id=name, type=kind, label=name, params=params)
    flow.queue_added(item)
    return item


class Runner:
    @flow.queue_execution
    async def execute(self, item):
        return await item.handler()


async def ok():
    return {"success": True}


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,render_type", [
    ("i2v", "video_i2v"), ("first_last", "video_first_last"), ("ref2v", "video_ref2v"),
])
async def test_video_queue_handoffs_complete_only_after_postprocess(mode, render_type):
    manager = QueueManager()
    queued = []

    async def add_item(kind, label, params, **kwargs):
        item = QueueItem(id=f"job-{len(queued)}", type=kind, label=label, params=params)
        flow.queue_added(item)
        queued.append(item)
        return item

    async def build_prompt(params, **kwargs):
        return {"success": True, "h3_prompt": "visible action"}

    async def render_video(params, **kwargs):
        await kwargs["progress_callback"](2, 4)
        return {"success": True, "mode": mode, "postprocess_job": {"spool_id": "test"}}

    async def postprocess(params, **kwargs):
        await kwargs["progress_callback"]({"percentage": 60, "phase": "encoding"})
        return {"success": True, "backup_name": "result.avif"}

    manager.add_item = add_item
    manager.video_mode = SimpleNamespace(
        build_prompt=build_prompt, render_video=render_video, postprocess_staged_video=postprocess,
    )
    root = await add_item("video_prompt_build", f"{mode} request", {"mode": mode})
    illustration = job("separate illustration", "illustration_llm_build")
    await manager._execute_item(root)
    assert flow.snapshot(kind="video")["status"] == "processing"
    assert queued[1].type == render_type
    await manager._execute_item(queued[1])
    assert flow.snapshot(kind="video")["status"] == "processing"
    assert queued[2].type == "video_postprocess"
    await manager._execute_item(queued[2])

    graph = flow.snapshot(kind="video")
    assert graph["status"] == "completed"
    nodes = {n["id"]: n for n in graph["nodes"]}
    assert nodes[queued[1].id]["dependencies"] == [root.id]
    assert nodes[queued[2].id]["dependencies"] == [queued[1].id]
    assert nodes[queued[1].id]["executor"] == "comfy"
    assert nodes[queued[2].id]["executor"] == "process"
    assert nodes[queued[2].id]["progress"] == 100
    result = next(n for n in graph["nodes"] if n["kind"] == "result")
    assert result["dependencies"] == [queued[2].id]
    assert flow.detail(graph["id"], result["id"])["output"]["backup_name"] == "result.avif"
    assert all("input" not in n and "output" not in n for n in graph["nodes"])
    assert flow.snapshot()["label"] == illustration.label
    assert flow.detail(flow.snapshot()["id"], illustration.id) is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", ["failed", "cancelled"])
async def test_video_terminal_state_after_handoff(terminal):
    root = job("video request")
    children = []

    async def prepare():
        children.append(job("render", "video_i2v"))
        return await ok()

    root.handler = prepare
    await Runner().execute(root)
    child = children[0]
    if terminal == "failed":
        async def fail():
            raise RuntimeError("render unavailable")
        child.handler = fail
        with pytest.raises(RuntimeError, match="render unavailable"):
            await Runner().execute(child)
    else:
        child.status = "cancelled"
        child.error = "queue cancellation"
        flow.queue_sync([child])
    graph = flow.snapshot(kind="video")
    assert graph["status"] == terminal
    assert not any(n["kind"] == "result" for n in graph["nodes"])
    assert flow.detail(graph["id"], child.id)["error"]


@pytest.mark.asyncio
async def test_video_routed_calls_preserve_parallel_dependencies_and_fallback(monkeypatch):
    async def dispatch(slot, messages, **kwargs):
        await llm_service._fire_queue_gate_acquired()
        await asyncio.sleep(0)
        return "[LLM 실패] first provider" if slot == "llm1" else messages[0]["content"]

    monkeypatch.setattr(llm_service, "_call_routed_text_slot", dispatch)
    seen_events = []

    async def call(name):
        context = llm_service.create_llm_execution_context("video_prompt_i2v_compose", call_name=name)
        return await llm_service.callLLMTask(
            context.task_key, [{"role": "user", "content": name}],
            execution_context=context, execution_observer=seen_events.append,
        )

    root = job("parallel video")
    async def build():
        await flow.gather(call("candidate A"), call("candidate B"), call("candidate C"))
        assert await call("selection") == "selection"
        return await ok()

    root.handler = build
    await Runner().execute(root)
    graph = flow.snapshot(kind="video")
    nodes = {n["label"]: n for n in graph["nodes"]}
    candidates = [nodes[f"candidate {name}"] for name in "ABC"]
    assert all(n["dependencies"] == [root.id] for n in candidates)
    assert set(nodes["selection"]["dependencies"]) == {n["id"] for n in candidates}
    detail = flow.detail(graph["id"], nodes["selection"]["id"])
    assert detail["status"] == "completed"
    assert detail["model"] == "fallback-test"
    assert detail["service"] == "fallback-provider"
    assert detail["llm_slot"] == "llm2"
    assert detail["phase"] == "fallback"
    assert detail["input"][0]["content"] == detail["output"] == "selection"
    assert any(e["type"] == "attempt_failure" for e in detail["attempts"])
    assert any(e["type"] == "execution_complete" for e in seen_events)


@pytest.mark.asyncio
async def test_video_routed_terminal_failure_is_visible(monkeypatch):
    async def dispatch(*args, **kwargs):
        return "[LLM 실패] unavailable"
    monkeypatch.setattr(llm_service, "_call_routed_text_slot", dispatch)
    root = job("failed prompt")
    async def build():
        result = await llm_service.callLLMTaskResult("video_prompt_i2v_compose", [])
        return {"success": result.accepted, "error": result.reason}
    root.handler = build
    await Runner().execute(root)
    graph = flow.snapshot(kind="video")
    llm = next(n for n in graph["nodes"] if n["kind"] == "llm")
    assert graph["status"] == llm["status"] == "failed"
    assert "unavailable" in flow.detail(graph["id"], llm["id"])["output"]


@pytest.mark.asyncio
async def test_graphs_publish_independently_and_old_video_cannot_replace_latest():
    events = []
    async def notify(kind, graph):
        events.append((kind, graph["label"]))
    flow.configure(notify)
    old = job("old video")
    new = job("new video")
    input_edit = job("new input", "video_instruction_refine", session_id="input-session")
    illustration = job("illustration", "illustration_llm_build")
    old.handler = new.handler = input_edit.handler = illustration.handler = ok
    await Runner().execute(new)
    await Runner().execute(input_edit)
    await Runner().execute(illustration)
    await Runner().execute(old)
    await asyncio.sleep(0.08)
    assert flow.snapshot(kind="video")["label"] == "new video"
    assert set(events) == {
        ("video_flow", "new video"),
        ("video_input_flow", "영상 입력 개선"),
        ("illustration_flow", "illustration"),
    }


@pytest.mark.asyncio
async def test_video_input_versions_append_human_choices_within_one_session():
    session_id = "video-input-session-a"
    flow.record_video_input_event(
        session_id,
        "session_start",
        input={"reference": {"kind": "backup", "name": "scene-a.png"}},
        output={"instruction": "사람이 직접 쓴 원문"},
    )
    draft = job("초안 만들기", "video_instruction_draft", session_id=session_id)
    draft.handler = ok
    await Runner().execute(draft)
    refine = job("입력 다듬기", "video_instruction_refine", session_id=session_id)
    refine.handler = ok
    await Runner().execute(refine)
    restored = flow.record_video_input_event(
        session_id,
        "restore_original",
        input={"instruction": "확장된 연출"},
        output={"instruction": "사람이 직접 쓴 원문"},
    )
    direct = job("다듬기 방향 명령하기", "video_instruction_direct", session_id=session_id)
    direct.handler = ok
    await Runner().execute(direct)
    applied = flow.record_video_input_event(
        session_id,
        "apply_direction",
        input={"instruction": "사람이 직접 쓴 원문", "candidate_instruction": "느린 동작"},
        output={"instruction": "느린 동작"},
    )

    graph = flow.snapshot(kind="video_input")
    assert graph["id"] == restored["id"] == applied["id"]
    assert graph["session_id"] == session_id
    nodes = {node["label"]: node for node in graph["nodes"]}
    assert nodes["초안 만들기"]["dependencies"] == [nodes["영상 입력 시작"]["id"]]
    assert nodes["입력 다듬기"]["dependencies"] == [draft.id]
    assert nodes["원문으로 되돌리기"]["dependencies"] == [refine.id]
    assert nodes["다듬기 방향 명령하기"]["dependencies"] == [nodes["원문으로 되돌리기"]["id"]]
    assert nodes["방향 수정안 적용"]["dependencies"] == [direct.id]
    restored_detail = flow.detail(graph["id"], nodes["원문으로 되돌리기"]["id"])
    assert restored_detail["input"]["instruction"] == "확장된 연출"
    assert restored_detail["output"]["instruction"] == "사람이 직접 쓴 원문"
    assert restored_detail["executor"] == "human"
    assert flow.snapshot(kind="video") is None


@pytest.mark.parametrize(
    "action,label",
    [
        ("restore_original", "원문으로 되돌리기"),
        ("apply_direction", "방향 수정안 적용"),
        ("discard_direction", "방향 수정안 폐기"),
    ],
)
def test_each_human_video_input_choice_is_a_distinct_node(action, label):
    session_id = f"session-{action}"
    first = flow.record_video_input_event(session_id, "session_start", output={"instruction": "원문"})
    graph = flow.record_video_input_event(
        session_id,
        action,
        input={"instruction": "수정안"},
        output={"instruction": "원문"},
    )
    nodes = {node["label"]: node for node in graph["nodes"]}
    assert nodes[label]["dependencies"] == [first["nodes"][0]["id"]]


def test_new_video_input_session_starts_a_fresh_graph():
    old = flow.record_video_input_event(
        "session-old",
        "session_start",
        input={"reference": "old-work"},
    )
    flow.record_video_input_event(
        "session-old",
        "restore_original",
        input={"instruction": "old edit"},
        output={"instruction": "old original"},
    )
    new = flow.record_video_input_event(
        "session-new",
        "session_start",
        input={"reference": "different-work"},
    )
    assert new["id"] != old["id"]
    assert new["session_id"] == "session-new"
    assert [node["label"] for node in new["nodes"]] == ["영상 입력 시작"]


@pytest.mark.asyncio
async def test_unrelated_queue_work_does_not_enter_video_graph():
    root = job("video")
    async def build():
        unrelated = job("other task", "llm_test")
        assert not hasattr(unrelated, "_illustration_flow")
        unrelated.handler = ok
        await Runner().execute(unrelated)
        return await ok()
    root.handler = build
    await Runner().execute(root)
    assert not any(n["label"] == "other task" for n in flow.snapshot(kind="video")["nodes"])


@pytest.mark.asyncio
async def test_vision_and_translation_join_before_prompt(monkeypatch):
    async def vision(*args, **kwargs):
        await llm_service._fire_queue_gate_acquired()
        return "reference context"
    async def text(*args, **kwargs):
        await llm_service._fire_queue_gate_acquired()
        return "translated direction"
    monkeypatch.setattr(llm_service, "callLLMVision", vision)
    monkeypatch.setattr(llm_service, "_call_routed_text_slot", text)
    async def call(name, fn):
        context = llm_service.create_llm_execution_context("video_prompt_i2v_compose", call_name=name)
        return await fn(context.task_key, [], execution_context=context)
    root = job("translated video")
    async def build():
        translation = flow.create_task(call("translation", llm_service.callLLMTask))
        await call("vision", llm_service.callLLMVisionTask)
        await translation
        flow.merge([translation])
        await call("compose", llm_service.callLLMTask)
        return await ok()
    root.handler = build
    await Runner().execute(root)
    nodes = {n["label"]: n for n in flow.snapshot(kind="video")["nodes"]}
    assert nodes["vision"]["dependencies"] == nodes["translation"]["dependencies"] == [root.id]
    assert set(nodes["compose"]["dependencies"]) == {nodes["vision"]["id"], nodes["translation"]["id"]}


@pytest.mark.asyncio
async def test_routed_observer_does_not_duplicate_illustration_llm_nodes(monkeypatch):
    async def text(*args, **kwargs):
        return "illustration result"
    monkeypatch.setattr(llm_service, "_call_routed_text_slot", text)
    root = job("illustration request", "illustration_llm_build")
    @flow.llm_call
    async def existing_call(name, messages):
        return await llm_service.callLLMTask("video_prompt_i2v_compose", messages)
    async def build():
        await existing_call("existing illustration call", [])
        return await ok()
    root.handler = build
    await Runner().execute(root)
    assert len([n for n in flow.snapshot()["nodes"] if n["kind"] == "llm"]) == 1
    assert flow.snapshot(kind="video") is None


@pytest.mark.asyncio
async def test_flow_api_selects_each_kind_and_fetches_details_without_server_startup():
    source = (Path(__file__).resolve().parents[1] / "server.py").read_text(encoding="utf-8-sig")
    handler = next(n for n in ast.parse(source).body if isinstance(n, ast.AsyncFunctionDef) and n.name == "handle_illustration_flow")
    namespace = {"web": web, "illustration_flow": flow}
    exec(compile(ast.Module(body=[handler], type_ignores=[]), "flow_api", "exec"), namespace)
    api = namespace[handler.name]
    illustration = job("illustration", "illustration_llm_build")
    video = job("video")
    input_flow = flow.record_video_input_event(
        "api-input-session",
        "session_start",
        input={"reference": "sample"},
    )
    input_node_id = input_flow["nodes"][0]["id"]
    for kind, expected_label, node_id in [
        ("illustration", illustration.label, illustration.id),
        ("video", video.label, video.id),
        ("video_input", "영상 입력 개선", input_node_id),
    ]:
        response = await api(SimpleNamespace(query={"kind": kind}))
        assert response.status == 200
        assert response.headers["Cache-Control"] == "no-store"
        graph = json.loads(response.text)["flow"]
        assert graph["label"] == expected_label
        detail = await api(SimpleNamespace(query={"run": graph["id"], "node": node_id}))
        assert json.loads(detail.text)["node"]["id"] == node_id
    missing = await api(SimpleNamespace(query={"run": "missing", "node": "missing"}))
    assert missing.status == 404


@pytest.mark.asyncio
async def test_human_video_input_event_api_appends_and_rejects_a_stale_session():
    source = (Path(__file__).resolve().parents[1] / "server.py").read_text(encoding="utf-8-sig")
    hip = next(
        node for node in ast.parse(source).body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "handle_video_input_flow_event"
    )
    namespace = {"web": web, "json": json, "traceback": traceback, "illustration_flow": flow}
    exec(compile(ast.Module(body=[hip], type_ignores=[]), "video_input_event_api", "exec"), namespace)
    api = namespace[hip.name]

    class Request:
        def __init__(self, body):
            self.body = body

        async def json(self):
            return self.body

    started = await api(Request({
        "session_id": "event-session",
        "action": "session_start",
        "input": {"reference": "scene"},
        "output": {"instruction": "원문"},
    }))
    restored = await api(Request({
        "session_id": "event-session",
        "action": "restore_original",
        "input": {"instruction": "수정문"},
        "output": {"instruction": "원문"},
    }))
    stale = await api(Request({
        "session_id": "another-session",
        "action": "restore_original",
        "input": {"instruction": "수정문"},
        "output": {"instruction": "원문"},
    }))

    assert started.status == restored.status == 200
    assert [node["label"] for node in json.loads(restored.text)["flow"]["nodes"]] == [
        "영상 입력 시작",
        "원문으로 되돌리기",
    ]
    assert stale.status == 409


def test_frontend_exposes_separate_video_input_tab_and_records_human_versions():
    root = Path(__file__).resolve().parents[1]
    flow_source = (root / "frontend" / "illustration_flow.js").read_text(encoding="utf-8-sig")
    frontend_source = (root / "frontend" / "index.html").read_text(encoding="utf-8-sig")

    assert "video_input: '영상 입력 개선'" in flow_source
    assert "window.receiveVideoInputFlow" in flow_source
    assert "case 'video_input_flow':" in frontend_source
    assert "video_input_session_id: inputFlowSessionId" in frontend_source
    assert "'restore_original'" in frontend_source
    assert "'apply_direction'" in frontend_source
    assert "'discard_direction'" in frontend_source
