"""Saved-server-RAW whole-generation Stop protocol; no image or network I/O."""
import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import illustration_flow as flow
import queue_manager as queue_module
import server
from modes import illustration_context_pipeline as pipeline


class Request:
    def __init__(self, *, body=None, batch_id="", prompt_id="", query=None):
        self.body = body
        self.match_info = {"batch_id": batch_id, "prompt_id": prompt_id}
        self.query = query or {}

    async def json(self):
        return self.body


def content(response):
    return json.loads(response.body)


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setattr(flow, "_raw_batches", {})
    monkeypatch.setattr(flow, "_latest", None)
    monkeypatch.setattr(flow, "_notify", None)
    monkeypatch.setattr(server, "prompts", {})
    # Never write session metadata or touch a real image during these tests.
    monkeypatch.setattr(pipeline, "set_session_regenerate_cancelled", lambda *a, **kw: None)
    manager = queue_module.QueueManager()

    async def no_work(*args, **kwargs):
        pass

    monkeypatch.setattr(manager, "_notify_queue_updated", no_work)
    monkeypatch.setattr(manager, "_process_loop", no_work)
    monkeypatch.setattr(server, "queue_manager", manager)
    return manager


def admit(token="a" * 24, session="risu_" + "1" * 64, index=0):
    run, reason = flow.admit_raw_batch(token, session, index)
    assert run is not None, reason
    return run


def queue_image(manager, run, *, name="pending", prompt_id="pending-prompt", item_type="illustration"):
    job = queue_module.QueueItem(
        id=name,
        type=item_type,
        label=name,
        params={
            "raw_batch_id": run["raw_batch_id"],
            "raw_batch_session_id": run["raw_batch_session_id"],
            "raw_batch_slot": 0,
            "prompt_id": prompt_id,
        },
    )
    job.completion_future = asyncio.get_running_loop().create_future()
    job.completion_future.add_done_callback(manager._mark_completion_future_observed)
    flow.queue_added(job)
    manager.items.append(job)
    if prompt_id:
        server.prompts[prompt_id] = {
            "status": "running", "prompt": {}, "outputs": {},
            "save_node_id": "9", "raw_batch_id": run["raw_batch_id"],
        }
    return job


@pytest.mark.asyncio
async def test_saved_raw_batch_stop_between_slots_blocks_late_requests_but_not_new_batch(env):
    first = admit()
    assert first["raw_batch_next_index"] == 1
    assert flow.admit_raw_batch(first["raw_batch_id"], first["raw_batch_session_id"], 2)[0] is None
    assert flow.admit_raw_batch(first["raw_batch_id"], "another-session", 1)[0] is None
    assert flow.request_cancel(first["id"])
    flow.finish_cancelled_raw_batch(first["raw_batch_id"])
    assert flow.admit_raw_batch(first["raw_batch_id"], first["raw_batch_session_id"], 1)[0] is None
    assert flow.raw_batch(first["raw_batch_id"])["status"] == "cancelled"
    other = admit("b" * 24, first["raw_batch_session_id"])
    assert other["raw_batch_next_index"] == 1
    assert other["cancel_requested"] is False
    assert flow.snapshot(other)["raw_batch_id"] == "b" * 24


@pytest.mark.asyncio
async def test_stop_settles_pending_proxy_history_with_specific_reason(env):
    run = admit()
    item = queue_image(env, run)
    response = await server.handle_illustration_flow_cancel(Request(body={"run": run["id"]}))
    assert response.status == 200
    assert content(response)["pending_cancelled"] == 1
    assert item.status == "cancelled"
    with pytest.raises(RuntimeError, match=flow.RAW_BATCH_CANCEL_REASON):
        await item.completion_future
    assert server.prompts["pending-prompt"]["status"] == "cancelled"
    history = content(await server.handle_history(Request(prompt_id="pending-prompt")))
    assert history["pending-prompt"]["status"]["status_str"] == "error"
    assert flow.RAW_BATCH_CANCEL_REASON in history["pending-prompt"]["status"]["messages"][0][1]["exception_message"]
    assert run["status"] == "cancelled"


@pytest.mark.asyncio
async def test_stop_keeps_active_image_result_and_cancels_only_pending_hybrid(env, monkeypatch):
    run = admit()
    active = queue_image(env, run, name="active", prompt_id="active-prompt")
    pending = queue_image(env, run, name="hybrid", prompt_id="", item_type="regenerate")
    entered, release = asyncio.Event(), asyncio.Event()

    async def execute(_self, _item):
        entered.set()
        await release.wait()
        server.prompts["active-prompt"]["status"] = "completed"
        return {"image": "finished"}

    monkeypatch.setattr(env, "_execute_item", flow.queue_execution(execute).__get__(env))
    task = asyncio.create_task(env._run_item_pipeline(active, is_gpu=False))
    await asyncio.wait_for(entered.wait(), 1)
    result = await env.cancel_illustration_flow(run["id"])
    assert result["active_finishing"] == 1
    assert pending.status == "cancelled"
    assert active.completion_future.done() is False
    assert run["status"] == "cancelling"
    release.set()
    await asyncio.wait_for(task, 1)
    assert active.status == "completed"
    assert active.completion_future.result() == {"image": "finished"}
    assert server.prompts["active-prompt"]["status"] == "completed"
    assert run["status"] == "cancelled"


@pytest.mark.asyncio
async def test_stop_during_worker_preparation_never_starts_image(env, monkeypatch):
    run = admit()
    job = queue_image(env, run, name="claimed")
    preparing, release = asyncio.Event(), asyncio.Event()
    executed = []

    async def prepare(*args):
        preparing.set()
        await release.wait()

    async def execute(*args):
        executed.append(True)
        return {}

    async def no_wait():
        pass

    monkeypatch.setattr(env, "prepare_local_gpu_target", prepare)
    monkeypatch.setattr(env, "_execute_item", execute)
    monkeypatch.setattr(env, "_wait_after_illustration", no_wait)
    task = asyncio.create_task(env._run_item_pipeline(job, is_gpu=True))
    await asyncio.wait_for(preparing.wait(), 1)
    result = await server.handle_illustration_flow_cancel(Request(body={"run": run["id"]}))
    assert content(result)["pending_cancelled"] == 1
    assert server.prompts["pending-prompt"]["status"] == "cancelled"
    release.set()
    await asyncio.wait_for(task, 1)
    assert executed == []
    assert job.status == "cancelled"
    assert run["status"] == "cancelled"


@pytest.mark.asyncio
async def test_late_queue_admission_is_rejected_without_affecting_unrelated_session(env):
    first = admit("c" * 24, "risu_" + "1" * 64)
    unrelated = admit("d" * 24, "risu_" + "2" * 64)
    unrelated_item = queue_image(env, unrelated, name="other", prompt_id="other-prompt")

    response = await server.handle_illustration_flow_cancel(Request(body={"run": first["id"]}))
    assert response.status == 200
    item = await env.add_item("illustration", "late", {
        "raw_batch_id": first["raw_batch_id"],
        "raw_batch_session_id": first["raw_batch_session_id"],
        "raw_batch_slot": 0,
    })
    assert item.status == "cancelled"
    with pytest.raises(RuntimeError, match=flow.RAW_BATCH_CANCEL_REASON):
        await item.completion_future
    assert unrelated_item.status == "pending"
    assert unrelated["cancel_requested"] is False
    assert unrelated["raw_batch_state"] == "active"
    status = await server.handle_raw_batch_status(Request(batch_id=unrelated["raw_batch_id"]))
    assert content(status)["state"] == "active"


@pytest.mark.asyncio
async def test_abort_cancels_pending_request_with_module_failure_reason(env):
    run = admit()
    pending = queue_image(env, run)
    result = await server.handle_raw_batch_status(Request(batch_id=run["raw_batch_id"], query={"a": "1"}))
    assert content(result)["state"] == "cancelled"
    assert "모듈" in content(result)["error"]
    with pytest.raises(RuntimeError, match="모듈"):
        await pending.completion_future
    assert "모듈" in server.prompts["pending-prompt"]["error"]
    assert flow.admit_raw_batch(run["raw_batch_id"], run["raw_batch_session_id"], 1)[0] is None


@pytest.mark.asyncio
async def test_prepare_arbitrates_stop_before_final_markup_commit(env):
    run = admit()
    stopped = await server.handle_raw_batch_status(Request(batch_id=run["raw_batch_id"], query={"p": "1"}))
    assert content(stopped)["state"] == "finalizing"
    assert flow.request_cancel(run["id"]) is False
    finished = await server.handle_raw_batch_status(Request(batch_id=run["raw_batch_id"], query={"f": "1"}))
    assert content(finished)["state"] == "completed"
    assert run["status"] == "completed"
