"""CALL1 and selection join before the assigned, parallel, ordered PLAN."""

import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

import illustration_flow as flow
from modes import illustration_context_pipeline as pipeline
from modes import illustration_scene_pipeline as scenes


def _data(content, heading):
    return json.JSONDecoder().raw_decode(content.split(heading + "\n", 1)[1])[0]


def _selected(name, count):
    return [{"anchor_segment": f"C{index:03d}", "characters": [name]}
            for index in range(1, count + 1)]


def _planned(selected):
    return [{**scene, "scene_brief":
             f"{scene['characters'][0]} performs visible action {int(scene['anchor_segment'][1:])}."}
            for scene in selected]


def _detail(batch):
    return "<lb-xnai>\n" + json.dumps({"scenes": [
        {"slot": scene["slot"], "camera": "medium shot",
         "characters": [{"name": character["name"], "positive": "reaching, coat",
                         "negative": "", "position": "center"}
                        for character in scene["characters"]],
         "scene": scene["brief"], "supplement": "connected hand and object"}
        for scene in batch]}) + "\n</lb-xnai>"


async def _build(name, count, **kwargs):
    toggles = {"output_count_min": count, "output_count_max": count,
               "key_visual": False, "call3_enabled": False, **kwargs.pop("toggles", {})}
    return await pipeline.build_from_context(
        {"session_id": "parallel-contract", "chats": [{"role": "char", "data":
            "\n\n".join(f"{name} performs visible action {i}." for i in range(1, count + 2))}]},
        toggles, f"### {name}\n-Name\n{name}\n-Appearance\nblue hair",
        pre_resolved_profile_result={"current_characters": [{"name": name}]}, **kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize(("name", "count", "limit", "enabled", "front_first"), [
    ("Mira", 7, 3, True, "CALL2-SELECT"), ("Ren", 6, 3, True, "CALL1"),
    ("Nora", 1, 8, True, "CALL2-SELECT"), ("Dae", 4, 1, True, "CALL1"),
    ("Yuna", 4, 3, False, "CALL2-SELECT"),
])
async def test_selection_precedes_simultaneous_plan_batches_and_reverse_completion_keeps_order(
    monkeypatch, name, count, limit, enabled, front_first,
):
    selected = _selected(name, count)
    expected_batches = min(count, limit) if enabled else 1
    entered = asyncio.Event()
    finished = {index: asyncio.Event() for index in range(1, expected_batches + 2)}
    finished[expected_batches + 1].set()
    starts, completions, calls, ready, notifications = [], [], [], [], []
    records = []
    front_started, front_completions = set(), []
    front_entered, first_finished = asyncio.Event(), asyncio.Event()
    active = maximum_active = 0
    monkeypatch.setattr(flow, "_latest", None)
    monkeypatch.setattr(flow, "_notify", None)
    monkeypatch.setattr(pipeline.lighbd_service, "_log_lighbd_history", records.append)

    async def fake_call(task_key, messages, **kwargs):
        nonlocal active, maximum_active
        metadata = pipeline.llm_service._stream_metadata_ctx.get({})
        stage = metadata["call_name"]
        calls.append(stage)
        content = messages[-1]["content"]
        if stage in {"CALL1", "CALL2-SELECT"}:
            front_started.add(stage)
            if len(front_started) == 2:
                front_entered.set()
            await asyncio.wait_for(front_entered.wait(), 3)
            if stage != front_first:
                await asyncio.wait_for(first_finished.wait(), 3)
        if stage == "CALL1":
            output = json.dumps({"SLOT0": {name: {"outfit": "coat", "hair": "ponytail"}}})
        elif stage == "CALL2-SELECT":
            assert task_key == "illustration_call2_select"
            assert "FULL CURRENT CANDIDATE SEGMENTS" in content
            assert "Each item contains only anchor_segment and characters" in messages[0]["content"]
            assert "CURRENT TRACKED OUTFIT AND HAIRSTYLE" not in content
            assert "PREVIOUS SAVED OUTFIT AND HAIRSTYLE (BEFORE CURRENT" in content
            output = json.dumps({"scene_plan": selected})
        elif stage.startswith("CALL2-PLAN"):
            assert task_key == "illustration_call2_plan"
            assert calls.count("CALL2-SELECT") == 1
            assert set(front_completions) == {"CALL1", "CALL2-SELECT"}
            assert _data(content, "# CURRENT TRACKED OUTFIT AND HAIRSTYLE (READ-ONLY)")["SLOT0"][name] == {
                "outfit": "coat", "hair": "ponytail"}
            assert "CALL2-AUDIT" not in calls
            batch = _data(content, "# ASSIGNED SELECTED SCENES (COUNT/ORDER/ANCHORS/CHARACTERS FIXED)")["scene_plan"]
            assert all(set(scene) == {"anchor_segment", "characters"} for scene in batch)
            assert "Choose one source-supported visible instant within each assigned anchor" in content
            assert _data(content, "# SINGLE SUBJECTS") == [name]
            index = len(starts) + 1
            starts.append(deepcopy(batch))
            active += 1
            maximum_active = max(maximum_active, active)
            if len(starts) == expected_batches:
                entered.set()
            await asyncio.wait_for(entered.wait(), 3)
            await asyncio.wait_for(finished[index + 1].wait(), 3)
            completions.append(index)
            active -= 1
            finished[index].set()
            output = json.dumps({"scene_plan": [
                {**scene, "scene_brief": scene["scene_brief"] + " Expanded pose and contact geometry."}
                for scene in _planned(batch)]})
        elif stage == "CALL2-DETAIL":
            assert "CALL2-AUDIT" not in calls
            output = _detail(_data(content, "# ONLY THIS BATCH'S PREPARED SCENES")["scenes"])
        else:
            raise AssertionError(stage)
        validator = kwargs.get("result_validator")
        if validator:
            assert validator(output) == (True, "")
        if stage in {"CALL1", "CALL2-SELECT"}:
            front_completions.append(stage)
            if stage == front_first:
                first_finished.set()
        return output

    async def notify(event):
        notifications.append(deepcopy(event))

    async def callback(data):
        ready.extend(data["items"])

    monkeypatch.setattr(pipeline.llm_service, "callLLMTask", fake_call)
    job = SimpleNamespace(id="parallel-contract", label="parallel-contract",
                          type="illustration_llm_build", params={}, status="pending")
    flow.queue_added(job)

    class Runner:
        @flow.queue_execution
        async def execute(self, item):
            return await _build(name, count, toggles={
                "call2_parallel_enabled": enabled, "call2_parallel_max_concurrency": limit},
                stream_notify=notify, on_call2_ready=callback, first_pass_single_v5=True)

    result = await Runner().execute(job)
    assert front_completions == [front_first, "CALL1" if front_first == "CALL2-SELECT" else "CALL2-SELECT"]
    assert maximum_active == expected_batches
    assert completions == list(range(expected_batches, 0, -1))
    assert max(map(len, starts)) - min(map(len, starts)) <= 1
    assert [scene for batch in starts for scene in batch] == selected
    assert json.loads(result["call2_select_output"])["scene_plan"] == selected
    assert len(result["call2_plan_outputs"]) == expected_batches
    assert len(json.loads(result["call2_plan_output"])["scene_plan"]) == count
    assert [item["slot"] for item in result["items"]] == list(range(count))
    assert sorted(item["slot"] for item in ready) == list(range(count))
    assert all("Expanded pose" in item["scene_brief"] for item in result["items"])
    plan_events = [event for event in notifications if event["call_name"].startswith("CALL2-PLAN")]
    assert {event["queue_subtask"]["index"] for event in plan_events} == set(range(1, expected_batches + 1))
    assert all(event["queue_subtask"]["total"] == expected_batches for event in plan_events)
    selection_record = next(record for record in records if record["call_name"] == "CALL2-SELECT")
    assert selection_record["task_key"] == "illustration_call2_select"
    assert selection_record["input"] and selection_record["output"]
    assert selection_record["status"] == "ok"
    assert selection_record["execution_id"] in result["llm_trace"]
    graph = flow.snapshot()
    nodes = {node["label"]: node for node in graph["nodes"]}
    plan_nodes = [node for node in graph["nodes"] if node["label"].startswith("CALL2-PLAN")]
    assert len(plan_nodes) == expected_batches
    assert nodes["CALL1"]["dependencies"] == nodes["CALL2-SELECT"]["dependencies"]
    assert all(set(node["dependencies"]) == {nodes["CALL1"]["id"], nodes["CALL2-SELECT"]["id"]}
               for node in plan_nodes)
    assert "CALL2-AUDIT" not in nodes
    detail_nodes = [node for node in graph["nodes"] if node["label"] == "CALL2-DETAIL"]
    plan_ids = {node["id"] for node in plan_nodes}
    assert all(len(node["dependencies"]) == 1 and set(node["dependencies"]) <= plan_ids for node in detail_nodes)


@pytest.mark.asyncio
@pytest.mark.parametrize(("name", "count", "plan_enabled", "plan_limit", "detail_enabled", "detail_limit"), [
    ("Mira", 7, True, 3, False, 5),
    ("Ren", 6, False, 4, True, 3),
    ("Noel", 7, True, 2, True, 4),
    ("Yuna", 4, False, 5, False, 6),
    ("Iris", 1, True, 7, True, 9),
    ("Lena", 5, True, 1, True, 3),
])
async def test_plan_and_detail_execute_with_independent_batch_counts_and_concurrency(
    monkeypatch, name, count, plan_enabled, plan_limit, detail_enabled, detail_limit,
):
    expected = {"plan": min(count, plan_limit) if plan_enabled else 1,
                "detail": min(count, detail_limit) if detail_enabled else 1}
    expected["detail"] = max(expected["detail"], expected["plan"])
    detail_cap = detail_limit if detail_enabled else 1
    barriers = {stage: asyncio.Event() for stage in expected}
    batches = {stage: [] for stage in expected}
    active = {stage: 0 for stage in expected}
    maximum = {stage: 0 for stage in expected}
    completed = {stage: 0 for stage in expected}
    ready = []

    async def fake_call(stage, messages, *_args, **_kwargs):
        content = messages[-1]["content"]
        if stage == "CALL1":
            return json.dumps({"SLOT0": {name: {"outfit": "coat", "hair": "ponytail"}}})
        if stage == "CALL2-SELECT":
            return json.dumps({"scene_plan": _selected(name, count)})
        if stage.startswith("CALL2-PLAN"):
            family = "plan"
            batch = _data(content, "# ASSIGNED SELECTED SCENES (COUNT/ORDER/ANCHORS/CHARACTERS FIXED)")["scene_plan"]
            output = json.dumps({"scene_plan": _planned(batch)})
        else:
            assert stage == "CALL2-DETAIL"
            family = "detail"
            batch = _data(content, "# ONLY THIS BATCH'S PREPARED SCENES")["scenes"]
            output = _detail(batch)
        batches[family].append(deepcopy(batch))
        active[family] += 1
        maximum[family] = max(maximum[family], active[family])
        if len(batches[family]) == (expected[family] if family == "plan" else min(detail_cap, expected[family])):
            barriers[family].set()
        try:
            await asyncio.wait_for(barriers[family].wait(), 3)
            completed[family] += 1
            return output
        finally:
            active[family] -= 1

    async def callback(data):
        ready.extend(data["items"])

    monkeypatch.setattr(pipeline, "_call_pipeline_llm", fake_call)
    result = await _build(name, count, toggles={
        "call2_parallel_enabled": plan_enabled, "call2_parallel_max_concurrency": plan_limit,
        "call2_detail_parallel_enabled": detail_enabled,
        "call2_detail_parallel_max_concurrency": detail_limit,
    }, on_call2_ready=callback)
    for family in expected:
        assert len(batches[family]) == completed[family] == expected[family]
        assert maximum[family] == (expected[family] if family == "plan" else min(detail_cap, expected[family]))
        assert sum(map(len, batches[family])) == count
        sizes = list(map(len, batches[family]))
        if family == "plan":
            assert max(sizes) - min(sizes) <= 1
        else:
            parents = [{scene["anchor_segment"] for scene in batch} for batch in batches["plan"]]
            assert all(any({scene["anchor_segment"] for scene in batch} <= parent for parent in parents)
                       for batch in batches["detail"])
        assert active[family] == 0
    assert len(result["call2_plan_outputs"]) == expected["plan"]
    assert [item["slot"] for item in result["items"]] == list(range(count))
    assert sorted(item["slot"] for item in ready) == list(range(count))


@pytest.mark.parametrize("change", ["anchor", "characters", "order", "missing"])
def test_plan_expansion_cannot_change_assigned_selection(change):
    assigned = _selected("Iris", 2)
    actual = _planned(assigned)
    if change == "anchor":
        actual[1]["anchor_segment"] = "C003"
    elif change == "characters":
        actual[1]["characters"] = ["Kian"]
    elif change == "order":
        actual.reverse()
    else:
        actual.pop()
    with pytest.raises(ValueError):
        scenes.parse_plan(json.dumps({"scene_plan": actual}),
            segments={f"C{i:03d}": {} for i in range(1, 4)},
            slot_map={f"C{i:03d}": i - 1 for i in range(1, 4)}, candidates=[0, 1, 2],
            names=["Iris", "Kian"], toggles={"output_count_min": 3, "output_count_max": 3,
                                           "character_limit": 2}, assigned_scenes=assigned)


@pytest.mark.asyncio
async def test_invalid_selection_stops_before_any_plan(monkeypatch):
    calls = []

    async def fake_call(stage, _messages, *_args, **_kwargs):
        calls.append(stage)
        if stage == "CALL1":
            return '{"SLOT0":{"Lio":{"outfit":"coat","hair":"short hair"}}}'
        assert stage == "CALL2-SELECT"
        return '{"scene_plan":[]}'

    monkeypatch.setattr(pipeline, "_call_pipeline_llm", fake_call)
    with pytest.raises(ValueError, match="CALL2-SELECT"):
        await _build("Lio", 2)
    assert calls == ["CALL1", "CALL2-SELECT"]


@pytest.mark.asyncio
async def test_cancel_parallel_plan_drains_every_active_batch(monkeypatch):
    entered = asyncio.Event()
    active, cancelled, calls = set(), set(), []

    async def fake_call(stage, messages, *_args, **_kwargs):
        calls.append(stage)
        if stage == "CALL1":
            return '{"SLOT0":{"Sora":{"outfit":"coat","hair":"short hair"}}}'
        if stage == "CALL2-SELECT":
            return json.dumps({"scene_plan": _selected("Sora", 3)})
        assert stage.startswith("CALL2-PLAN")
        active.add(stage)
        if len(active) == 3:
            entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.add(stage)
            raise
        finally:
            active.remove(stage)

    monkeypatch.setattr(pipeline, "_call_pipeline_llm", fake_call)
    task = asyncio.create_task(_build("Sora", 3))
    try:
        await asyncio.wait_for(entered.wait(), 3)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert not active
    assert len(cancelled) == 3
    assert "CALL2-AUDIT" not in calls and "CALL2-DETAIL" not in calls


@pytest.mark.asyncio
async def test_plan_failure_stops_without_synthesizing_a_brief(monkeypatch):
    calls = []

    async def fake_call(stage, _messages, *_args, **_kwargs):
        calls.append(stage)
        if stage == "CALL1":
            return '{"SLOT0":{"Tao":{"outfit":"coat","hair":"short hair"}}}'
        if stage == "CALL2-SELECT":
            return json.dumps({"scene_plan": _selected("Tao", 2)})
        assert stage.startswith("CALL2-PLAN")
        raise RuntimeError("configured plan route unavailable")

    monkeypatch.setattr(pipeline, "_call_pipeline_llm", fake_call)
    with pytest.raises(RuntimeError, match="configured plan route unavailable"):
        await _build("Tao", 2)
    assert "CALL2-AUDIT" not in calls and "CALL2-DETAIL" not in calls


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_stage", ["CALL1", "CALL2-SELECT"])
@pytest.mark.parametrize("failure", ["exception", "invalid_output"])
@pytest.mark.parametrize("legacy_disabled", [False, True])
async def test_front_stage_failure_cancels_other_call_and_stops_before_plan(
    monkeypatch, failed_stage, failure, legacy_disabled,
):
    entered = asyncio.Event()
    active, cancelled, calls = set(), set(), []

    async def fake_call(stage, _messages, *_args, **_kwargs):
        calls.append(stage)
        assert stage in {"CALL1", "CALL2-SELECT"}
        active.add(stage)
        if len(active) == 2:
            entered.set()
        try:
            await asyncio.wait_for(entered.wait(), 3)
            if stage == failed_stage:
                if failure == "exception":
                    raise RuntimeError(f"{stage} routed failure")
                return "{}"
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.add(stage)
            raise
        finally:
            active.remove(stage)

    monkeypatch.setattr(pipeline, "_call_pipeline_llm", fake_call)
    with pytest.raises(RuntimeError if failure == "exception" else ValueError, match=failed_stage):
        await _build("Haru", 2, toggles={"call1_enabled": not legacy_disabled})
    assert set(calls) == {"CALL1", "CALL2-SELECT"}
    assert cancelled == ({"CALL1", "CALL2-SELECT"} - {failed_stage})
    assert not active


@pytest.mark.asyncio
async def test_cancelling_front_stage_drains_call1_and_select(monkeypatch):
    entered = asyncio.Event()
    active, cancelled = set(), set()

    async def fake_call(stage, _messages, *_args, **_kwargs):
        assert stage in {"CALL1", "CALL2-SELECT"}
        active.add(stage)
        if len(active) == 2:
            entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.add(stage)
            raise
        finally:
            active.remove(stage)

    monkeypatch.setattr(pipeline, "_call_pipeline_llm", fake_call)
    task = asyncio.create_task(_build("Lena", 2))
    try:
        await asyncio.wait_for(entered.wait(), 3)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert cancelled == {"CALL1", "CALL2-SELECT"}
    assert not active


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["Eli", "Noel"])
async def test_legacy_call1_off_still_tracks_current_events_and_saves_state(monkeypatch, name):
    calls = []
    saved = ["coat buttoned; tied hair", "coat opened; other details unchanged"]
    current_event = {"outfit": "coat, open coat", "hair": "hair down"}

    async def fake_call(stage, messages, *_args, **_kwargs):
        calls.append(stage)
        content = messages[-1]["content"]
        if stage == "CALL1":
            assert _data(content, "# PREVIOUS SAVED CALL1 STATE (RAW CHRONOLOGICAL DETAIL MEMORY; APPLY NARRATIVE CHANGES)") == {name: saved}
            return json.dumps({"SLOT0": {name: current_event}})
        if stage == "CALL2-SELECT":
            assert _data(content, "# PREVIOUS SAVED OUTFIT AND HAIRSTYLE (BEFORE CURRENT; READ-ONLY)") == {name: saved}
            return json.dumps({"scene_plan": _selected(name, 1)})
        if stage == "CALL2-PLAN":
            tracking = _data(content, "# CURRENT TRACKED OUTFIT AND HAIRSTYLE (READ-ONLY)")
            assert tracking == {"SLOT0": {name: current_event}}
            return json.dumps({"scene_plan": _planned(_selected(name, 1))})
        assert stage == "CALL2-DETAIL"
        batch = _data(content, "# ONLY THIS BATCH'S PREPARED SCENES")["scenes"]
        assert batch[0]["characters"][0]["state"] == current_event
        assert "anchor_event" not in batch[0]["characters"][0]
        return _detail(batch)

    monkeypatch.setattr(pipeline, "_call_pipeline_llm", fake_call)
    result = await _build(name, 1, toggles={"call1_enabled": False}, history_plan={
        "state_before": {name.lower(): {"canonical_name": name, "call1_status_history": saved}}})
    assert calls == ["CALL1", "CALL2-SELECT", "CALL2-PLAN", "CALL2-DETAIL"]
    assert result["call1_output"]
    assert [json.loads(status) for status in result["character_states_after"][name.lower()]["call1_status_history"]] == [current_event]
