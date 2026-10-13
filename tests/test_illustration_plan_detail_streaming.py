"""PLAN winners feed DETAIL immediately, with shared caps and one-time effects."""
import asyncio
from collections import Counter
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

import illustration_flow as flow
from modes import illustration_context_pipeline as pipeline


def data(messages, heading):
    return json.JSONDecoder().raw_decode(messages[-1]["content"].split(heading + "\n", 1)[1])[0]


def planned(batch):
    return {"scene_plan": [{**scene, "scene_brief": "A visible pause at this story instant."} for scene in batch]}


def detail(batch):
    return "<lb-xnai>\n" + json.dumps({"scenes": [
        {"slot": scene["slot"], "camera": "medium shot", "scene": "quiet room", "supplement": "",
         "characters": [{"name": character["name"], "positive": character["state"]["outfit"],
                         "negative": "", "position": "center"} for character in scene["characters"]]}
        for scene in batch]}) + "\n</lb-xnai>"


async def build(name, count, *, plan_limit=2, detail_limit=2, toggles=None, **kwargs):
    return await pipeline.build_from_context(
        {"session_id": "plan-detail-streaming", "chats": [{"role": "char", "data":
            "\n\n".join(f"{name} pauses at moment {index}." for index in range(count + 1))}]},
        {"output_count_min": count, "output_count_max": count, "key_visual": False,
         "call3_enabled": False, "call2_parallel_enabled": True,
         "call2_parallel_max_concurrency": plan_limit, "call2_detail_parallel_enabled": True,
         "call2_detail_parallel_max_concurrency": detail_limit, **(toggles or {})},
        f"### {name}\n-Appearance\nshort hair", pre_resolved_profile_result={
            "current_characters": [{"name": name}]}, **kwargs)


def front(stage, messages, name, count):
    if stage == "CALL1":
        return json.dumps({"SLOT0": {name: {"outfit": "nude", "hair": "short hair"}},
                           "SLOT2": {name: {"outfit": "white shirt"}}})
    if stage == "CALL2-SELECT":
        return json.dumps({"scene_plan": [{"anchor_segment": f"C{index + 1:03d}", "characters": [name]}
                                         for index in range(count)]})
    return None


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["Shoko", "Mira"])
async def test_fast_plan_starts_detail_and_registers_images_before_slow_plan_finishes(monkeypatch, name):
    slow_started, release_slow, first_ready = (asyncio.Event() for _ in range(3))
    calls, ready, detail_inputs = [], [], []

    @flow.llm_call
    async def llm(stage, messages, *_args, **_kwargs):
        calls.append(stage)
        output = front(stage, messages, name, 4)
        if output is not None:
            return output
        if stage.startswith("CALL2-PLAN"):
            batch = data(messages, "# ASSIGNED SELECTED SCENES (COUNT/ORDER/ANCHORS/CHARACTERS FIXED)")["scene_plan"]
            if batch[0]["anchor_segment"] == "C003":
                slow_started.set()
                await release_slow.wait()
            else:
                await slow_started.wait()
            return json.dumps(planned(batch))
        assert stage == "CALL2-DETAIL"
        batch = data(messages, "# ONLY THIS BATCH'S PREPARED SCENES")["scenes"]
        detail_inputs.extend(deepcopy(batch))
        return detail(batch)

    async def register(value):
        ready.extend(value["items"])
        if value["items"][0]["slot"] == 0:
            assert not release_slow.is_set()
            first_ready.set()

    monkeypatch.setattr(pipeline, "_call_pipeline_llm", llm)
    item = SimpleNamespace(id="plan-detail-stream-qa", label="streaming", type="illustration_llm_build", params={}, status="pending")
    flow.queue_added(item)

    class Runner:
        @flow.queue_execution
        async def execute(self, item):
            return await build(name, 4, on_call2_ready=register)

    task = asyncio.create_task(Runner().execute(item))
    try:
        await asyncio.wait_for(first_ready.wait(), 3)
        assert not task.done() and {entry["slot"] for entry in ready} <= {0, 1}
        release_slow.set()
        result = await asyncio.wait_for(task, 3)
    finally:
        release_slow.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert "CALL2-AUDIT" not in calls
    assert [item["slot"] for item in result["items"]] == [0, 1, 2, 3]
    assert Counter(item["slot"] for item in ready) == Counter(range(4))
    inputs = {scene["slot"]: scene["characters"][0]["state"] for scene in detail_inputs}
    assert inputs[0]["outfit"] == inputs[1]["outfit"] == "nude"
    assert inputs[2]["outfit"] == inputs[3]["outfit"] == "white shirt"
    assert "call2_audit_output" not in result and "call2_audited_plan" not in result
    nodes = flow.snapshot()["nodes"]
    plans = {node["id"] for node in nodes if node["label"].startswith("CALL2-PLAN")}
    details = [node for node in nodes if node["label"] == "CALL2-DETAIL"]
    assert len(details) == 2 and all(len(node["dependencies"]) == 1 and set(node["dependencies"]) <= plans for node in details)


@pytest.mark.asyncio
@pytest.mark.parametrize("name,count,plan_limit", [("Ren", 9, 3), ("Soyeon", 8, 4)])
async def test_detail_limit_is_shared_across_all_completed_plan_groups(monkeypatch, name, count, plan_limit):
    saturated, release = asyncio.Event(), asyncio.Event()
    active = maximum = started = 0

    async def llm(stage, messages, *_args, **_kwargs):
        nonlocal active, maximum, started
        output = front(stage, messages, name, count)
        if output is not None:
            return output
        if stage.startswith("CALL2-PLAN"):
            return json.dumps(planned(data(messages, "# ASSIGNED SELECTED SCENES (COUNT/ORDER/ANCHORS/CHARACTERS FIXED)")["scene_plan"]))
        assert stage == "CALL2-DETAIL"
        started += 1
        active += 1
        maximum = max(maximum, active)
        if active == 2:
            saturated.set()
        try:
            await release.wait()
            return detail(data(messages, "# ONLY THIS BATCH'S PREPARED SCENES")["scenes"])
        finally:
            active -= 1

    monkeypatch.setattr(pipeline, "_call_pipeline_llm", llm)
    task = asyncio.create_task(build(name, count, plan_limit=plan_limit, detail_limit=2))
    try:
        await asyncio.wait_for(saturated.wait(), 3)
        assert active == started == maximum == 2
        release.set()
        result = await asyncio.wait_for(task, 3)
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert maximum == 2 and active == 0
    assert [item["slot"] for item in result["items"]] == list(range(count))


@pytest.mark.asyncio
async def test_cancellation_drains_running_detail_and_other_plan_callbacks(monkeypatch):
    entered = asyncio.Event()
    active, cancelled = set(), []

    async def llm(stage, messages, *_args, **_kwargs):
        output = front(stage, messages, "Nora", 6)
        if output is not None:
            return output
        if stage.startswith("CALL2-PLAN"):
            return json.dumps(planned(data(messages, "# ASSIGNED SELECTED SCENES (COUNT/ORDER/ANCHORS/CHARACTERS FIXED)")["scene_plan"]))
        assert stage == "CALL2-DETAIL"
        slots = tuple(scene["slot"] for scene in data(messages, "# ONLY THIS BATCH'S PREPARED SCENES")["scenes"])
        active.add(slots)
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.append(slots)
            raise
        finally:
            active.remove(slots)

    monkeypatch.setattr(pipeline, "_call_pipeline_llm", llm)
    task = asyncio.create_task(build("Nora", 6, plan_limit=3, detail_limit=1))
    await asyncio.wait_for(entered.wait(), 3)
    assert len(active) == 1
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not active and len(cancelled) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["Ari", "Jin"])
async def test_losing_plan_retry_cannot_start_duplicate_detail_or_image_registration(monkeypatch, name):
    cancelled = []
    registrations = []
    monkeypatch.setattr(pipeline.lighbd_service, "_update_lighbd_history_records", lambda _updates: True)

    async def llm(stage, messages, *_args, **_kwargs):
        output = front(stage, messages, name, 4)
        if output is not None:
            return output
        if stage.startswith("CALL2-PLAN"):
            batch = data(messages, "# ASSIGNED SELECTED SCENES (COUNT/ORDER/ANCHORS/CHARACTERS FIXED)")["scene_plan"]
            if batch[0]["anchor_segment"] == "C003" and "[느리다고? 다시해!]" not in stage:
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    cancelled.append(stage)
                    raise
            return json.dumps(planned(batch))
        assert stage == "CALL2-DETAIL"
        return detail(data(messages, "# ONLY THIS BATCH'S PREPARED SCENES")["scenes"])

    async def register(value):
        registrations.extend(item["slot"] for item in value["items"])

    monkeypatch.setattr(pipeline, "_call_pipeline_llm", llm)
    result = await asyncio.wait_for(build(name, 4, toggles={
        "call2_parallel_slow_retry_enabled": True, "call2_parallel_slow_retry_remaining": 1,
        "call2_parallel_slow_retry_progress_threshold": 99}, on_call2_ready=register), 3)
    assert cancelled and Counter(registrations) == Counter(range(4))
    assert len(result["call2_detail_outputs"]) == 2
