"""DETAIL tail races accept one response before masks and early image dispatch."""

import asyncio
from collections import Counter
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

import illustration_flow as flow
from modes import illustration_context_pipeline as pipeline


def _selected(name, count):
    return [{"anchor_segment": f"C{index:03d}", "characters": [name],
             "scene_brief": f"{name} reaches toward object {index}."}
            for index in range(1, count + 1)]


def _batch(messages):
    return json.JSONDecoder().raw_decode(messages[-1]["content"].split(
        "# ONLY THIS BATCH'S PREPARED SCENES\n", 1)[1])[0]["scenes"]


def _detail(batch, marker="accepted"):
    return "<lb-xnai>\n" + json.dumps({"scenes": [
        {"slot": scene["slot"], "camera": "medium shot",
         "characters": [{"name": character["name"], "positive": "reaching, coat",
                         "negative": "", "position": "center"}
                        for character in scene["characters"]],
         "scene": scene["brief"], "supplement": marker} for scene in batch]}) + "\n</lb-xnai>"


async def _build(name, count, **kwargs):
    toggles = {"output_count_min": count, "output_count_max": count,
               "key_visual": False, "speak_enabled": False,
               "call2_parallel_enabled": False,
               "call2_detail_parallel_enabled": True,
               **kwargs.pop("toggles", {})}
    return await pipeline.build_from_context(
        {"session_id": "detail-race-contract", "chats": [{"role": "char", "data":
            "\n\n".join(f"{name} reaches toward object {index}." for index in range(1, count + 2))}]},
        toggles, f"### {name}\n-Name\n{name}\n-Appearance\nblue hair",
        pre_resolved_profile_result={"current_characters": [{"name": name}]}, **kwargs)


def _front(stage, name, count):
    if stage == "CALL1":
        return json.dumps({"SLOT0": {name: {"outfit": "coat", "hair": "blue hair"}}})
    if stage in {"CALL2-SELECT", "CALL2-PLAN"}:
        return json.dumps({"scene_plan": _selected(name, count)})
    raise AssertionError(stage)


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["Mira", "Noel"])
@pytest.mark.parametrize("winner", ["duplicate", "primary", "invalid_duplicate"])
async def test_detail_race_routes_logs_and_dispatches_only_valid_winner(monkeypatch, name, winner):
    count = 5
    entered, duplicate_started = asyncio.Event(), asyncio.Event()
    starts, cancelled, notifications, records, updates, masks, ready = [], [], [], [], [], [], []
    active = maximum = 0
    monkeypatch.setattr(flow, "_latest", None)
    monkeypatch.setattr(flow, "_notify", None)
    monkeypatch.setattr(pipeline.lighbd_service, "_log_lighbd_history", records.append)
    monkeypatch.setattr(pipeline.lighbd_service, "_update_lighbd_history_records", updates.append)

    async def llm_stub(task_key, messages, **kwargs):
        nonlocal active, maximum
        stage = pipeline.llm_service._stream_metadata_ctx.get()["call_name"]
        if stage == "MULTI-CHAR-MASK":
            return "mask ready"
        if not stage.startswith("CALL2-DETAIL"):
            return _front(stage, name, count)
        assert task_key == "illustration_call2"
        batch = _batch(messages)
        index = batch[0]["slot"] // 2 + 1
        duplicate = "[" in stage
        starts.append((index, duplicate))
        active += 1
        maximum = max(maximum, active)
        observer = kwargs["stream_observer"]
        observer({"type": "request_mode", "streaming": False})
        if sum(not is_duplicate for _, is_duplicate in starts) == 3:
            entered.set()
        try:
            await asyncio.wait_for(entered.wait(), 3)
            if index == 3:
                if duplicate:
                    duplicate_started.set()
                    if winner == "primary":
                        await asyncio.Event().wait()
                    if winner == "invalid_duplicate":
                        invalid = deepcopy(batch)
                        invalid[0]["characters"][0]["name"] = "Unassigned person"
                        ok, reason = kwargs["result_validator"](_detail(invalid))
                        assert not ok
                        raise ValueError(reason)
                elif winner == "duplicate":
                    await asyncio.Event().wait()
                else:
                    await asyncio.wait_for(duplicate_started.wait(), 3)
            output = _detail(batch, "duplicate" if duplicate else "primary")
            assert kwargs["result_validator"](output) == (True, "")
            await kwargs["execution_observer"]({"type": "attempt_success", "phase": "primary",
                "slot": "llm2", "attempt": 1, "total_attempts": 1, "attempt_id": stage})
            kwargs["metadata_sink"].update(completion_tokens=50, prompt_tokens=80)
            return output
        except asyncio.CancelledError:
            cancelled.append((index, duplicate))
            raise
        finally:
            active -= 1

    async def layout_stub(descriptors, *_args, **_kwargs):
        slots = [descriptor["slot"] for descriptor in descriptors]
        masks.append(slots)
        await pipeline._call_pipeline_llm("MULTI-CHAR-MASK", [
            {"role": "user", "content": json.dumps(slots)}])
        for descriptor in descriptors:
            descriptor["multi_char_layout"] = {"ready": True}

    async def notify(event):
        notifications.append(deepcopy(event))

    async def dispatch(data):
        assert all(item.get("multi_char_layout", {}).get("ready") for item in data["items"])
        ready.extend(deepcopy(data["items"]))

    monkeypatch.setattr(pipeline.llm_service, "callLLMTask", llm_stub)
    monkeypatch.setattr(pipeline, "calculate_multi_char_layouts", layout_stub)
    job = SimpleNamespace(id="detail-race-contract", label="detail-race-contract",
                          type="illustration_llm_build", params={}, status="pending")
    flow.queue_added(job)

    class Runner:
        @flow.queue_execution
        async def execute(self, _job):
            return await _build(name, count, toggles={
                "call2_parallel_slow_retry_enabled": False,
                "call2_detail_parallel_max_concurrency": 3,
                "call2_detail_parallel_slow_retry_enabled": True,
            }, stream_notify=notify, on_call2_ready=dispatch, enable_multi_char_layout=True)

    result = await asyncio.wait_for(Runner().execute(job), 5)
    assert maximum <= 3 and active == 0
    assert Counter(starts) == Counter([(1, False), (2, False), (3, False), (3, True)])
    assert Counter(item["slot"] for item in ready) == Counter(range(count))
    assert sorted(masks) == [[slot] for slot in range(count)]
    assert [item["slot"] for item in result["items"]] == list(range(count))
    expected = "duplicate" if winner == "duplicate" else "primary"
    assert result["items"][-1]["supplement"] == expected
    assert expected in result["call2_detail_outputs"][-1]
    if winner != "invalid_duplicate":
        assert cancelled == [(3, winner == "primary")]
    detail_records = [record for record in records if record["task_key"] == "illustration_call2"]
    assert len(detail_records) == 4
    assert all(record["execution_id"] in result["llm_trace"] for record in detail_records)
    assert all(record["input"] and "elapsed" in record for record in detail_records)
    successes = [record for record in detail_records if record["status"] == "ok"]
    assert len(successes) == 3
    assert all(record["llm_slot"] == "llm2" and record["completion_tokens"] == 50 for record in successes)
    if winner == "invalid_duplicate":
        assert any(record["status"] == "error" and record["error"] for record in detail_records)
    assert {value["race_outcome"] for update in updates for value in update.values()} == {"winner", "loser"}
    detail_events = [event for event in notifications if event["call_name"].startswith("CALL2-DETAIL")]
    assert {event["queue_subtask"]["index"] for event in detail_events} == {1, 2, 3}
    assert all(event["queue_subtask"]["total"] == 3 for event in detail_events)
    graph = flow.snapshot()
    mask = next(node for node in graph["nodes"] if node["label"] == "MULTI-CHAR-MASK"
                and flow.detail(graph["id"], node["id"])["input"][0]["content"] == "[4]")
    winner_id = next(history_id for update in updates for history_id, value in update.items()
                     if value["race_outcome"] == "winner")
    winner_node = next(node for node in graph["nodes"] if node.get("execution_id") == winner_id)
    assert mask["dependencies"] == [winner_node["id"]]


@pytest.mark.asyncio
@pytest.mark.parametrize(("name", "count", "detail_options", "partial_length", "expected_duplicate"), [
    ("Ren", 3, {"enabled": False}, 0, False),
    ("Sora", 1, {"enabled": True}, 0, False),
    ("Tao", 3, {"enabled": True, "progress_enabled": True}, 10000, False),
    ("Lena", 3, {"enabled": True, "progress_enabled": False, "tps_enabled": False}, 0, False),
    ("Eli", 3, {"enabled": True, "progress_enabled": False, "tps_enabled": True}, 0, True),
    ("Nora", 3, {"enabled": True, "progress_enabled": True, "tps_enabled": True,
                 "condition_operator": "and", "tps_threshold": 1000}, 400, False),
    ("Yuna", 3, {"enabled": True, "progress_enabled": True, "tps_enabled": True,
                 "condition_operator": "or", "tps_threshold": 1000}, 400, True),
])
async def test_detail_uses_own_conditions_and_fast_batches_dispatch_before_tail(
    monkeypatch, name, count, detail_options, partial_length, expected_duplicate,
):
    entered, release = asyncio.Event(), asyncio.Event()
    starts, ready = [], []
    monkeypatch.setattr(pipeline.lighbd_service, "_update_lighbd_history_records", lambda _updates: None)

    async def llm_stub(stage, messages, *_args, **kwargs):
        if not stage.startswith("CALL2-DETAIL"):
            return _front(stage, name, count)
        batch = _batch(messages)
        starts.append(stage)
        if len(starts) == count:
            entered.set()
        observer = kwargs.get("stream_observer")
        if observer:
            observer({"type": "request_mode", "streaming": True})
            observer({"type": "stream_open", "stream_id": stage,
                      "partial_length": partial_length})
        await asyncio.wait_for(entered.wait(), 3)
        if batch[0]["slot"] != count - 1 and "condition_operator" in detail_options:
            # Make TPS low while the streaming progress remains high, so AND
            # and OR must reach opposite decisions about the same tail request.
            await asyncio.sleep(0.2)
        if batch[0]["slot"] == count - 1:
            if "[" not in stage:
                await asyncio.wait_for(release.wait(), 3)
            else:
                release.set()
        return _detail(batch)

    async def dispatch(data):
        ready.extend(item["slot"] for item in data["items"])
        if not expected_duplicate and len(ready) >= count - 1:
            release.set()

    monkeypatch.setattr(pipeline, "_call_pipeline_llm", llm_stub)
    if count == 1:
        release.set()
    result = await asyncio.wait_for(_build(name, count, toggles={
        # PLAN's opposite settings must not determine DETAIL's race.
        "call2_parallel_slow_retry_enabled": not detail_options["enabled"],
        "call2_parallel_slow_retry_progress_enabled": False,
        "call2_parallel_slow_retry_tps_enabled": False,
        "call2_detail_parallel_max_concurrency": 3,
        **{f"call2_detail_parallel_slow_retry_{key}": value for key, value in detail_options.items()},
    }, on_call2_ready=dispatch), 5)
    assert sum("[" in stage for stage in starts) == int(expected_duplicate)
    assert Counter(ready) == Counter(range(count))
    assert len(result["items"]) == count


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["cancel", "dispatch_error"])
async def test_detail_cancel_or_dispatch_failure_drains_requests_and_finalizers(monkeypatch, failure):
    pending_entered, finalizer_entered = asyncio.Event(), asyncio.Event()
    active, cancelled = set(), set()

    async def llm_stub(stage, messages, *_args, **_kwargs):
        if not stage.startswith("CALL2-DETAIL"):
            return _front(stage, "Haru", 2)
        batch = _batch(messages)
        if batch[0]["slot"] == 0:
            await asyncio.wait_for(pending_entered.wait(), 3)
            return _detail(batch)
        active.add("request")
        pending_entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.add("request")
            raise
        finally:
            active.remove("request")

    async def dispatch(_data):
        finalizer_entered.set()
        if failure == "dispatch_error":
            raise RuntimeError("image enqueue unavailable")
        active.add("finalizer")
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.add("finalizer")
            raise
        finally:
            active.remove("finalizer")

    monkeypatch.setattr(pipeline, "_call_pipeline_llm", llm_stub)
    task = asyncio.create_task(_build("Haru", 2, on_call2_ready=dispatch))
    try:
        await asyncio.wait_for(finalizer_entered.wait(), 3)
        if failure == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            with pytest.raises(RuntimeError, match="image enqueue unavailable"):
                await asyncio.wait_for(task, 3)
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    assert not active
    assert cancelled == ({"request", "finalizer"} if failure == "cancel" else {"request"})


@pytest.mark.asyncio
async def test_unusable_detail_preserves_original_error_and_cancels_remaining_request(monkeypatch):
    entered = asyncio.Event()
    cancelled = []

    async def llm_stub(stage, messages, *_args, **_kwargs):
        if not stage.startswith("CALL2-DETAIL"):
            return _front(stage, "Aster", 2)
        if _batch(messages)[0]["slot"] == 0:
            await asyncio.wait_for(entered.wait(), 3)
            return "unparseable DETAIL response"
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.append(1)
            raise

    monkeypatch.setattr(pipeline, "_call_pipeline_llm", llm_stub)
    with pytest.raises(ValueError, match="CALL2-DETAIL"):
        await asyncio.wait_for(_build("Aster", 2), 3)
    assert cancelled == [1]


@pytest.mark.asyncio
async def test_invalid_duplicate_and_valid_primary_with_identical_timestamps_accept_once(monkeypatch):
    duplicate_entered = asyncio.Event()
    accepted = []
    # Replace only this module's clock so asyncio's deadlines keep advancing.
    monkeypatch.setattr(pipeline, "time", SimpleNamespace(monotonic=lambda: 1.0))
    monkeypatch.setattr(pipeline.lighbd_service, "_update_lighbd_history_records", lambda _updates: None)

    async def invoke(_job, index, _total, attempt_kind, *_args):
        if index == 2:
            if attempt_kind == "duplicate":
                duplicate_entered.set()
                raise ValueError("duplicate does not contain assigned scenes")
            await asyncio.wait_for(duplicate_entered.wait(), 3)
        return {"raw": f"accepted batch {index}"}

    async def finalize(index, _value):
        accepted.append(index)

    result = await asyncio.wait_for(pipeline._run_parallel_pipeline_jobs(
        [{"weight": 1}, {"weight": 1}], group_id="detail-timestamp-tie",
        group_label="DETAIL timestamp tie", max_concurrency=2,
        slow_retry_enabled=True, slow_retry_remaining=1,
        slow_retry_progress_enabled=True, slow_retry_progress_threshold=50,
        slow_retry_tps_enabled=False, slow_retry_tps_threshold=5.0,
        slow_retry_condition_operator="and", stream_notify=None, invoke=invoke,
        on_resolved=finalize, fail_fast=True), 3)
    assert [value["raw"] for value in result] == ["accepted batch 1", "accepted batch 2"]
    assert Counter(accepted) == Counter([1, 2])
