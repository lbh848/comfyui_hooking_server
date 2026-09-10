import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import illustration_flow as flow


def test_flow_dialog_is_pinned_to_viewport_center():
    source = (Path(__file__).resolve().parents[1] / "frontend" / "illustration_flow.js").read_text(encoding="utf-8")
    assert ".if-modal{position:fixed;inset:0;z-index:2147483644;margin:auto;" in source


def test_flow_stays_below_memo_and_llm_layers():
    source = (Path(__file__).resolve().parents[1] / "frontend" / "illustration_flow.js").read_text(encoding="utf-8")
    assert ".if-layer-backdrop{position:fixed;inset:0;z-index:2147483643;" in source
    assert ".if-detail{z-index:2147483645}" in source
    assert "showModal()" not in source
    assert "flowBackdrop.hidden = false; modal.show();" in source


def test_flow_zoom_has_feedback_before_any_request():
    source = (Path(__file__).resolve().parents[1] / "frontend" / "illustration_flow.js").read_text(encoding="utf-8")
    assert "function updateZoomDisplay()" in source
    assert "현재 확대 비율 ${Math.round(scale * 100)}%" in source
    assert "updateZoomDisplay(); render();" in source


def test_flow_exposes_stop_button_and_cancel_endpoint():
    root = Path(__file__).resolve().parents[1]
    source = (root / "frontend" / "illustration_flow.js").read_text(encoding="utf-8")
    server_source = (root / "server.py").read_text(encoding="utf-8-sig")
    assert "stopButton = button('중단', cancelCurrentFlow);" in source
    assert "fetch('/api/illustration_flow/cancel'" in source
    assert "flow?.cancel_requested" in source
    assert "!terminal && cancelling ? '중단 중…' : '중단'" in source
    assert 'app.router.add_post("/api/illustration_flow/cancel", handle_illustration_flow_cancel)' in server_source

def test_flow_layout_reserves_columns_in_pipeline_order():
    source = (Path(__file__).resolve().parents[1] / "frontend" / "illustration_flow.js").read_text(encoding="utf-8")
    assert "label.startsWith('CALL1-BACKTRANSLATE')" in source
    assert "const isPlanAssetColumnLabel = value =>" in source
    assert "label === 'CALL2-PLAN'" in source
    assert "label === 'ORIGINAL-ASSET'" in source
    assert "? '__early_compact__'" in source
    assert "? '__call2_plan_asset__'" in source
    assert "if (group === '__early_compact__')" in source


def test_flow_layout_keeps_logical_stage_retries_and_images_in_one_column():
    source = (Path(__file__).resolve().parents[1] / "frontend" / "illustration_flow.js").read_text(encoding="utf-8")
    assert "String(n.layout_group || n.id)" in source
    assert "if (parentGroup !== group) dependencies.add(parentGroup)" in source
    assert "Retries/partial repairs in the same" in source
    assert "actual generation" in source


def test_pipeline_assigns_distinct_layout_groups_to_detail_and_authority_audit():
    root = Path(__file__).resolve().parents[1]
    source = (root / "modes" / "illustration_context_pipeline.py").read_text(encoding="utf-8")
    assert '"CALL2-DETAIL": ("call2_detail",' in source
    assert '"CALL2-AUTHORITY-AUDIT": (' in source
    assert '"call2_authority_audit",' in source
    assert "layout_group=(queue_subtask_group[0] if queue_subtask_group else task_key)" in source


def test_image_queue_and_generation_stage_use_separate_layout_groups():
    root = Path(__file__).resolve().parents[1]
    flow_source = (root / "illustration_flow.py").read_text(encoding="utf-8")
    server_source = (root / "server.py").read_text(encoding="utf-8")
    assert 'layout_group=None if is_root or is_review else "illustration_images"' in flow_source
    assert 'layout_group="illustration_generation"' in server_source


def test_flow_layout_honors_semantic_order_within_a_column():
    root = Path(__file__).resolve().parents[1]
    source = (root / "frontend" / "illustration_flow.js").read_text(encoding="utf-8")
    pipeline_source = (root / "modes" / "illustration_context_pipeline.py").read_text(encoding="utf-8")
    assert "const orderedNodes = [...nodes].sort((a, b) =>" in source
    assert "Number(a.layout_order)" in source
    assert '"call3": -100' in pipeline_source
    assert '"call3_correction": -90' in pipeline_source
    assert "layout_order=(" in pipeline_source


def test_flow_layout_keeps_each_column_top_aligned():
    source = (Path(__file__).resolve().parents[1] / "frontend" / "illustration_flow.js").read_text(encoding="utf-8")
    assert "y: 32 + lane * 136" in source
    assert "Keep every logical column top-aligned" in source
    assert "const childrenByParent = new Map();" not in source
    assert "desiredY: (top + bottom) / 2" not in source


def test_flow_nodes_use_executor_colored_backgrounds():
    source = (Path(__file__).resolve().parents[1] / "frontend" / "illustration_flow.js").read_text(encoding="utf-8")
    assert "const executorColors = {llm:" in source
    assert "background:color-mix(in srgb,var(--bg2,#172033) 68%,var(--node-tint,#94a3b8) 32%)" in source
    assert "card.style.setProperty('--node-tint', executorColors[executor] || executorColors.process)" in source
    assert "처리 주체" in source


def test_flow_detail_falls_back_to_error_and_latest_raw_response():
    source = (Path(__file__).resolve().parents[1] / "frontend" / "illustration_flow.js").read_text(encoding="utf-8")
    assert "function detailOutput(n)" in source
    assert "attempts[index].raw_response" in source
    assert "fallback.error = n.error" in source
    assert "const output = detailOutput(n);" in source


@pytest.mark.asyncio
async def test_flow_nodes_can_share_explicit_layout_group():
    job = item(); flow.queue_added(job)

    @flow.stage("image stage", layout_group="illustration_generation")
    async def image_stage():
        return True

    async def work():
        child = item("image child", "illustration")
        child.params = {"provider": "comfy"}
        flow.queue_added(child)
        async def child_work():
            await image_stage()
            return {}
        child.handler = child_work
        await Runner().execute(child)
        return {}

    job.handler = work
    await Runner().execute(job)
    graph = flow.snapshot()
    nodes = {n["label"]: n for n in graph["nodes"]}
    assert nodes["image child"]["layout_group"] == "illustration_images"
    assert nodes["image stage"]["layout_group"] == "illustration_generation"


@pytest.mark.asyncio
async def test_stage_executor_can_follow_actual_provider():
    job = item(); flow.queue_added(job)

    @flow.stage(
        "provider stage",
        executor=lambda inputs: "comfy" if inputs.get("provider") == "comfy" else "process",
    )
    async def provider_stage(provider="comfy"):
        return provider

    async def work():
        await provider_stage("comfy")
        await provider_stage("chansub")
        return {}

    job.handler = work
    await Runner().execute(job)
    nodes = [n for n in flow.snapshot()["nodes"] if n["label"] == "provider stage"]
    assert [n["executor"] for n in nodes] == ["comfy", "process"]


@pytest.mark.asyncio
async def test_llm_and_image_queue_nodes_expose_executor_metadata():
    job = item(); flow.queue_added(job)

    async def work():
        await llm("plan", [])
        child = item("slot", "illustration")
        child.params = {"provider": "comfy"}
        flow.queue_added(child)
        child.handler = lambda: None
        async def child_work():
            return {}
        child.handler = child_work
        await Runner().execute(child)
        return {}

    job.handler = work
    await Runner().execute(job)
    nodes = {n["label"]: n for n in flow.snapshot()["nodes"]}
    assert nodes["plan"]["executor"] == "llm"
    assert nodes["slot"]["executor"] == "comfy"
    assert nodes[job.label]["executor"] == "process"


@pytest.fixture(autouse=True)
def isolated_flow(monkeypatch):
    monkeypatch.setattr(flow, "_latest", None)
    monkeypatch.setattr(flow, "_notify", None)


def item(name="요청", item_type="illustration_llm_build"):
    return SimpleNamespace(id=name, label=name, type=item_type, params={}, status="pending")


class Runner:
    @flow.queue_execution
    async def execute(self, job):
        return await job.handler()


@flow.llm_call
async def llm(name, messages, *, entered=None, release=None):
    flow.llm_metadata(status="processing", model="test-model")
    if entered:
        entered.set()
    if release:
        await release.wait()
    return "출력: " + name


@pytest.mark.asyncio
@pytest.mark.parametrize("names", [("CALL1", "CALL2 A", "CALL2 B", "CALL3"), ("분석", "장면 하나", "장면 둘", "병합")])
async def test_actual_fork_join_and_details(names):
    job = item()
    flow.queue_added(job)
    async def work():
        await llm(names[0], [{"content": "원문"}])
        await flow.gather(llm(names[1], []), llm(names[2], []))
        await llm(names[3], [])
        return {"success": True}
    job.handler = work
    await Runner().execute(job)
    graph = flow.snapshot()
    nodes = {n["label"]: n for n in graph["nodes"]}
    assert nodes[names[1]]["dependencies"] == nodes[names[2]]["dependencies"] == [nodes[names[0]]["id"]]
    assert set(nodes[names[3]]["dependencies"]) == {nodes[names[1]]["id"], nodes[names[2]]["id"]}
    assert graph["status"] == "completed"
    assert "input" not in nodes[names[0]]
    assert flow.detail(graph["id"], nodes[names[0]]["id"])["input"] == [{"content": "원문"}]
    json.dumps(graph)


@pytest.mark.asyncio
async def test_serial_execution_keeps_chain():
    job = item()
    flow.queue_added(job)
    async def work():
        for name in ["first", "second", "third"]:
            await llm(name, [])
        return {}
    job.handler = work
    await Runner().execute(job)
    nodes = [n for n in flow.snapshot()["nodes"] if n["kind"] == "llm"]
    assert nodes[1]["dependencies"] == [nodes[0]["id"]]
    assert nodes[2]["dependencies"] == [nodes[1]["id"]]


@pytest.mark.asyncio
async def test_parallel_cap_shows_waiting_before_worker_starts():
    job = item(); flow.queue_added(job)
    entered, release = asyncio.Event(), asyncio.Event()
    gate = asyncio.Semaphore(1)
    async def worker(name):
        async with gate:
            return await llm(name, [], entered=entered, release=release)
    async def work():
        await flow.gather(*(flow.create_task(worker(name), flow_label=name) for name in ["one", "two", "three"]))
        return {}
    job.handler = work
    running = asyncio.create_task(Runner().execute(job))
    await entered.wait()
    nodes = {n["label"]: n for n in flow.snapshot()["nodes"]}
    assert nodes[job.label]["status"] == "processing"
    assert [nodes[n]["status"] for n in ["one", "two", "three"]] == ["processing", "waiting", "waiting"]
    release.set(); await running
    assert len([n for n in flow.snapshot()["nodes"] if n["kind"] == "llm"]) == 3


@pytest.mark.asyncio
async def test_older_run_cannot_replace_latest_or_leak_context():
    old = item("old"); flow.queue_added(old)
    entered, release = asyncio.Event(), asyncio.Event()
    async def work():
        await llm("old call", [], entered=entered, release=release)
        return {}
    old.handler = work
    running = asyncio.create_task(Runner().execute(old))
    await entered.wait()
    new = item("new"); flow.queue_added(new)
    latest = flow.snapshot()["id"]
    release.set(); await running
    assert flow.snapshot()["id"] == latest
    assert [n["label"] for n in flow.snapshot()["nodes"]] == ["new"]
    assert flow._run.get() is None


@pytest.mark.asyncio
async def test_child_queue_and_postprocess_join():
    job = item(); flow.queue_added(job)
    @flow.stage("image")
    async def image():
        return b"png", ""
    @flow.stage("postprocess")
    async def postprocess():
        return b"png"
    async def work():
        await llm("plan", [])
        child = item("slot", "illustration"); flow.queue_added(child)
        child.handler = image
        task = asyncio.create_task(Runner().execute(child))
        await llm("subtitle", [])
        await task
        flow.merge([child])
        await postprocess()
        return {}
    job.handler = work
    await Runner().execute(job)
    nodes = {n["label"]: n for n in flow.snapshot()["nodes"]}
    assert set(nodes["postprocess"]["dependencies"]) == {nodes["image"]["id"], nodes["subtitle"]["id"]}
    assert flow.detail(flow.snapshot()["id"], nodes["image"]["id"])["output"] == [{"image_bytes": 3}, ""]


@pytest.mark.asyncio
async def test_real_pipeline_fallback_metadata_and_raw_responses(monkeypatch):
    from modes import illustration_context_pipeline as pipeline
    from modes import llm_service, lighbd_service
    monkeypatch.setattr(lighbd_service, "_log_lighbd_history", lambda record: None)
    monkeypatch.setattr(llm_service, "_base_config_get", lambda key, default=None: {"llm_model2": "fallback-model", "llm_service2": "fallback-service"}.get(key, default))
    async def call(task_key, messages, **kwargs):
        observer = kwargs["execution_observer"]
        await observer({"type": "attempt_start", "phase": "primary", "slot": "llm1", "attempt": 1})
        flow.llm_metadata(status="processing")
        await observer({"type": "attempt_failure", "phase": "primary", "slot": "llm1", "raw_response": "bad response", "reason": "parse failed"})
        await observer({"type": "attempt_start", "phase": "fallback", "slot": "llm2", "attempt": 1})
        flow.llm_metadata(status="processing")
        await observer({"type": "attempt_success", "phase": "fallback", "slot": "llm2", "attempt": 1, "raw_response": "usable response"})
        return "usable response"
    monkeypatch.setattr(llm_service, "callLLMTask", call)
    job = item(); flow.queue_added(job)
    async def work():
        await pipeline._call_pipeline_llm("CALL1", [{"role": "user", "content": "test"}])
        return {}
    job.handler = work
    await Runner().execute(job)
    n = next(n for n in flow.snapshot()["nodes"] if n["kind"] == "llm")
    details = flow.detail(flow.snapshot()["id"], n["id"])
    assert details["model"] == "fallback-model"
    assert details["service"] == "fallback-service"
    assert details["output"] == "usable response"
    assert details["attempts"][1]["raw_response"] == "bad response"


@pytest.mark.asyncio
async def test_failed_llm_exposes_error_and_undecoded_raw_response():
    damaged_base64 = "YWJjA"

    @flow.llm_call
    async def failing_llm(name, messages):
        flow.llm_metadata(status="processing", model="test-model")
        flow.llm_attempt(
            {"type": "attempt_failure", "phase": "primary", "slot": "llm1",
             "raw_response": damaged_base64, "reason": "response parse failed"},
            "test-model",
            "test-service",
        )
        raise RuntimeError("response parse failed")

    job = item(); flow.queue_added(job)

    async def work():
        await failing_llm("CALL2-DETAIL", [{"role": "user", "content": "test"}])

    job.handler = work
    with pytest.raises(RuntimeError, match="response parse failed"):
        await Runner().execute(job)

    graph = flow.snapshot()
    node = next(n for n in graph["nodes"] if n["label"] == "CALL2-DETAIL")
    details = flow.detail(graph["id"], node["id"])
    assert details["status"] == "failed"
    assert details["output"] == {
        "error": "response parse failed",
        "raw_response": damaged_base64,
    }


@pytest.mark.asyncio
async def test_failed_non_llm_stage_exposes_error_as_output():
    @flow.stage("different failing stage")
    async def failing_stage(value):
        raise ValueError(f"cannot process {value}")

    job = item(); flow.queue_added(job)

    async def work():
        await failing_stage("scene")

    job.handler = work
    with pytest.raises(ValueError, match="cannot process scene"):
        await Runner().execute(job)

    graph = flow.snapshot()
    node = next(n for n in graph["nodes"] if n["label"] == "different failing stage")
    details = flow.detail(graph["id"], node["id"])
    assert details["output"] == {"error": "cannot process scene"}


@pytest.mark.asyncio
@pytest.mark.parametrize("bot", ["", "   ", None])
async def test_missing_bot_emits_warning(monkeypatch, bot):
    import server
    events = []
    async def notify(kind, data):
        events.append((kind, data))
    monkeypatch.setattr(server, "notify_frontend", notify)
    with pytest.raises(RuntimeError, match="활성 봇을 선택"):
        server._require_active_illustration_bot(bot, context="test request")
    await asyncio.sleep(0)
    assert events == [("illustration_warning", {"message": "삽화를 생성하려면 먼저 활성 봇을 선택해야 합니다.", "context": "test request"})]


@pytest.mark.asyncio
async def test_configured_bot_does_not_warn(monkeypatch):
    import server
    async def notify(*args):
        pytest.fail("configured bot must not warn")
    monkeypatch.setattr(server, "notify_frontend", notify)
    assert server._require_active_illustration_bot("봇 A", context="test") == "봇 A"
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_terminal_failure_and_cancelled_queue():
    job = item(); flow.queue_added(job)
    async def fail():
        raise RuntimeError("execution failed")
    job.handler = fail
    with pytest.raises(RuntimeError):
        await Runner().execute(job)
    assert flow.snapshot()["status"] == "failed"
    failed_graph = flow.snapshot()
    assert flow.detail(failed_graph["id"], job.id)["output"] == {"error": "execution failed"}
    cancelled = item("cancelled"); flow.queue_added(cancelled)
    cancelled.status = "cancelled"
    flow.queue_sync([cancelled])
    assert flow.snapshot()["status"] == "cancelled"
    cancelled_graph = flow.snapshot()
    assert flow.detail(cancelled_graph["id"], cancelled.id)["output"] == {"error": "작업 취소"}


@pytest.mark.asyncio
async def test_unrelated_queue_work_does_not_inherit_illustration():
    job = item(); flow.queue_added(job)
    async def other_work():
        assert flow._run.get() is None
        await llm("unrelated", [])
        return {}
    async def work():
        other = item("other", "tag_analysis"); other.handler = other_work
        await Runner().execute(other)
        await llm("related", [])
        return {}
    job.handler = work
    await Runner().execute(job)
    assert [n["label"] for n in flow.snapshot()["nodes"] if n["kind"] == "llm"] == ["related"]


@pytest.mark.asyncio
async def test_cancel_before_parallel_task_starts():
    job = item(); flow.queue_added(job)
    async def work():
        task = flow.create_task(llm("never started", []), flow_label="queued")
        task.cancel()
        await flow.gather(task, return_exceptions=True)
        return {}
    job.handler = work
    await Runner().execute(job)
    node = next(n for n in flow.snapshot()["nodes"] if n["label"] == "queued")
    assert node["status"] == "cancelled"
    assert flow.detail(flow.snapshot()["id"], node["id"])["output"] == {"error": "대기 작업 취소"}


@pytest.mark.asyncio
async def test_queue_returned_failure_updates_request_node():
    job = item(); flow.queue_added(job)
    async def work():
        return {"success": False, "error": "provider unavailable"}
    job.handler = work
    await Runner().execute(job)
    graph = flow.snapshot()
    assert graph["status"] == "failed"
    request = next(n for n in graph["nodes"] if n["id"] == job.id)
    assert request["status"] == "failed"
    assert request["error"] == "provider unavailable"


@pytest.mark.asyncio
async def test_cancel_request_marks_flow_and_late_children_cancelled():
    job = item()
    flow.queue_added(job)
    graph = flow.snapshot()
    assert flow.request_cancel(graph["id"]) is True
    assert flow.snapshot()["status"] == "cancelling"
    assert flow.snapshot()["cancel_requested"] is True

    token = flow._run.set(flow._latest)
    try:
        child = item("late child", "illustration")
        flow.queue_added(child)
    finally:
        flow._run.reset(token)

    assert child.status == "cancelled"
    assert getattr(child, "_illustration_cancelled_on_add", False) is True
    detail = flow.detail(graph["id"], child.id)
    assert detail["status"] == "cancelled"
    assert "중단" in detail["error"]


@pytest.mark.asyncio
async def test_queue_manager_allows_active_image_to_finish_after_flow_cancel(monkeypatch):
    import queue_manager as queue_module

    manager = queue_module.QueueManager()
    job = queue_module.QueueItem(
        id="active-flow",
        type="illustration",
        label="active-flow",
        params={},
    )
    job.completion_future = asyncio.get_running_loop().create_future()
    job.completion_future.add_done_callback(manager._mark_completion_future_observed)
    flow.queue_added(job)
    manager.items.append(job)
    entered = asyncio.Event()
    release = asyncio.Event()

    async def execute(_self, _item):
        entered.set()
        await release.wait()
        return {"success": True, "image": "finished"}

    wrapped = flow.queue_execution(execute).__get__(manager, type(manager))
    monkeypatch.setattr(manager, "_execute_item", wrapped)
    running = asyncio.create_task(manager._run_item_pipeline(job, is_gpu=False))
    await entered.wait()

    result = await manager.cancel_illustration_flow(flow.snapshot()["id"])

    assert result["active_finishing"] == 1
    assert result["active_cancelled"] == 0
    assert running.done() is False
    assert job.completion_future.done() is False

    release.set()
    await asyncio.wait_for(running, timeout=1)

    assert job.status == "completed"
    assert running.cancelled() is False
    assert flow.snapshot()["status"] == "cancelled"
    assert job.completion_future.result() == {"success": True, "image": "finished"}


@pytest.mark.asyncio
async def test_cancel_request_interrupts_active_llm_immediately():
    job = item("llm-root")
    flow.queue_added(job)
    entered = asyncio.Event()
    release = asyncio.Event()

    async def work():
        await llm("CALL2-DETAIL", [], entered=entered, release=release)
        return {"success": True}

    job.handler = work
    running = asyncio.create_task(Runner().execute(job))
    await entered.wait()
    run_id = flow.snapshot()["id"]

    assert flow.request_cancel(run_id) is True
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(running, timeout=1)

    graph = flow.snapshot()
    assert graph["status"] == "cancelled"
    node = next(n for n in graph["nodes"] if n["label"] == "CALL2-DETAIL")
    assert node["status"] == "cancelled"
    assert "중단" in node["error"] or "취소" in node["error"]


@pytest.mark.asyncio
async def test_queue_manager_rejects_late_child_added_after_flow_cancel():
    import queue_manager as queue_module

    manager = queue_module.QueueManager()
    root = item("root")
    flow.queue_added(root)
    run = flow._latest
    assert flow.request_cancel(run["id"]) is True

    token = flow._run.set(run)
    try:
        child = await manager.add_item(
            "illustration",
            "late queue child",
            {"provider": "comfy"},
        )
    finally:
        flow._run.reset(token)

    assert child.status == "cancelled"
    assert child in manager.items
    assert child.completion_future.done()
    assert isinstance(child.completion_future.exception(), RuntimeError)
    detail = flow.detail(run["id"], child.id)
    assert detail["status"] == "cancelled"
    assert "중단" in detail["error"]
