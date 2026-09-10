import asyncio
import base64
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import illustration_flow
import queue_manager as queue_module
import server
from modes import illustration_quality_inspection as quality
from modes import illustration_context_pipeline as pipeline
from modes import lighbd_service


class _Request:
    def __init__(self, method: str, body: dict | None = None, **match_info):
        self.method = method
        self._body = body
        self.match_info = match_info

    async def json(self):
        return self._body


def _entries() -> list[dict]:
    return [
        {
            "slot": 2,
            "backup_name": "river-scene-2",
            "prompt_id": "prompt-2",
            "image_bytes": b"\x89PNG\r\n\x1a\nraw-two",
            "positive": "Ari reaches toward the lantern",
            "negative": "extra unrelated people",
            "descriptor": {
                "kind": "scene",
                "scene": "Ari reaches toward the lantern.",
                "camera": "medium side view",
            },
        },
        {
            "slot": 5,
            "backup_name": "river-scene-5",
            "prompt_id": "prompt-5",
            "image_bytes": b"RIFF1234WEBPraw-five",
            "positive": "Ari opens the lantern beside the river",
            "negative": "unmotivated costume change",
            "descriptor": {
                "kind": "scene",
                "scene": "At the river, Ari opens the lantern.",
                "camera": "wide view",
            },
        },
    ]


def _inspection_config() -> dict:
    return {
        "llm_service": "provider-one",
        "llm_model": "model-one",
        "llm_service2": "provider-two",
        "llm_model2": "model-two",
    }


def _runtime_snapshot(*, inspection=True, original_assets=False) -> dict:
    return {
        "bot_name": "integration-bot",
        "illustration_quality_inspection_enabled": inspection,
        "provider": "comfy",
        "illustration_workflow_type": "v3_anima",
        "chansub_workflow_type": "anima",
        "clamp_enabled": False,
        "clamp_value": 1.2,
        "illustration_context_toggles": {
            "illustration_enabled": True,
            "original_asset_enabled": original_assets,
            "original_asset_count": 1 if original_assets else 0,
            "prompt_format": "v3",
            "multi_char_mask_enabled": False,
        },
        "word_rules": [],
    }


@pytest.mark.asyncio
async def test_quality_settings_post_preserves_unrelated_config_and_gets_new_value(monkeypatch):
    config = {
        "illustration_quality_inspection_enabled": False,
        "unrelated_setting": "keep-me",
        "nested": {"value": 7},
    }
    saved: list[dict] = []
    monkeypatch.setattr(server, "app_config", config)
    monkeypatch.setattr(server, "save_config", lambda candidate: saved.append(copy.deepcopy(candidate)))

    response = await server.handle_illustration_quality_inspection_settings(
        _Request("POST", {"enabled": True})
    )
    assert response.status == 200
    assert json.loads(response.text) == {"enabled": True}
    assert saved == [{
        "illustration_quality_inspection_enabled": True,
        "unrelated_setting": "keep-me",
        "nested": {"value": 7},
    }]
    assert config["unrelated_setting"] == "keep-me"
    assert config["nested"] == {"value": 7}

    get_response = await server.handle_illustration_quality_inspection_settings(
        _Request("GET")
    )
    assert get_response.status == 200
    assert json.loads(get_response.text) == {"enabled": True}


@pytest.mark.asyncio
async def test_quality_settings_failed_save_does_not_change_runtime_config(monkeypatch):
    config = {
        "illustration_quality_inspection_enabled": False,
        "unrelated_setting": "keep-me",
    }
    monkeypatch.setattr(server, "app_config", config)

    def fail_save(_candidate):
        raise OSError("settings disk is unavailable")

    monkeypatch.setattr(server, "save_config", fail_save)
    response = await server.handle_illustration_quality_inspection_settings(
        _Request("POST", {"enabled": True})
    )

    assert response.status == 500
    assert config == {
        "illustration_quality_inspection_enabled": False,
        "unrelated_setting": "keep-me",
    }


@pytest.mark.asyncio
async def test_quality_enqueue_is_nonblocking_and_releases_snapshot_after_completion(monkeypatch):
    loop = asyncio.get_running_loop()
    queue_item = SimpleNamespace(
        id="inspection-queue-1",
        completion_future=loop.create_future(),
    )
    captured: dict = {}
    live_snapshot: dict = {}

    async def fake_add_item(item_type, label, params, **kwargs):
        captured.update(item_type=item_type, label=label, params=params, kwargs=kwargs)
        captured["handler"] = kwargs["runtime_handler"]
        freevars = dict(zip(
            captured["handler"].__code__.co_freevars,
            (cell.cell_contents for cell in captured["handler"].__closure__ or ()),
        ))
        live_snapshot["entries"] = freevars["snapshot_entries"]
        return queue_item

    async def fake_execute(_queue_item, *, session_id, context, entries, flow_run_id):
        live_snapshot["seen"] = copy.deepcopy(entries)
        live_snapshot["context"] = context
        live_snapshot["flow_run_id"] = flow_run_id
        return {"images": [], "overall_feedback": "done"}

    monkeypatch.setattr(server.queue_manager, "add_item", fake_add_item)
    monkeypatch.setattr(server.illustration_quality_inspection, "execute_inspection", fake_execute)
    original = _entries()
    await server._enqueue_illustration_quality_inspection(
        session_id="session-queue",
        context="The original river narrative.",
        entries=original,
    )

    assert captured["item_type"] == quality.TASK_KEY
    assert captured["params"] == {"session_id": "session-queue", "slots": [2, 5]}
    assert "runtime_handler" in captured["kwargs"]
    assert "illustration_quality_inspection" in queue_module.LLM_TYPES
    assert live_snapshot.get("seen") is None

    original[0]["image_bytes"] = b"caller-mutated"
    await captured["handler"](queue_item)
    assert live_snapshot["seen"][0]["image_bytes"] == _entries()[0]["image_bytes"]
    assert live_snapshot["context"] == "The original river narrative."
    assert live_snapshot["entries"] == []

    queue_item.completion_future.set_result({"success": True})
    await asyncio.sleep(0)
    assert live_snapshot["entries"] == []


@pytest.mark.asyncio
async def test_quality_enqueue_releases_snapshot_when_waiting_item_is_cancelled(monkeypatch):
    loop = asyncio.get_running_loop()
    queue_item = SimpleNamespace(
        id="inspection-queue-cancelled",
        completion_future=loop.create_future(),
    )
    captured: dict = {}

    async def fake_add_item(_item_type, _label, _params, **kwargs):
        captured["handler"] = kwargs["runtime_handler"]
        return queue_item

    monkeypatch.setattr(server.queue_manager, "add_item", fake_add_item)
    await server._enqueue_illustration_quality_inspection(
        session_id="session-cancelled",
        context="A narrative that will not be inspected.",
        entries=_entries(),
    )
    freevars = dict(zip(
        captured["handler"].__code__.co_freevars,
        (cell.cell_contents for cell in captured["handler"].__closure__ or ()),
    ))
    snapshot = freevars["snapshot_entries"]
    assert snapshot

    queue_item.completion_future.cancel()
    await asyncio.sleep(0)
    assert snapshot == []


@pytest.mark.asyncio
async def test_quality_type_runs_through_unified_runtime_handler_and_background_flow_set():
    manager = queue_module.QueueManager()
    item = queue_module.QueueItem(
        id="runtime-inspection",
        type=quality.TASK_KEY,
        label="inspection",
        params={},
    )
    item._runtime_handler = lambda _item: asyncio.sleep(0, result={"success": True})

    assert quality.TASK_KEY in queue_module.LLM_TYPES
    assert quality.TASK_KEY in illustration_flow.BACKGROUND_LLM_TYPES
    assert await manager._handle_runtime_llm_task(item) == {"success": True}


@pytest.mark.asyncio
async def test_process_prompt_captures_raw_pixels_and_actual_prompt_before_backup(monkeypatch):
    prompt_id = "quality-raw-source"
    raw_pixels = b"raw-generated-pixels"
    captioned_pixels = b"captioned-backup-pixels"
    captured: dict = {}
    runtime = _runtime_snapshot()
    runtime["clamp_enabled"] = True
    runtime["clamp_value"] = 1.2
    server.prompts[prompt_id] = {
        "status": "running",
        "prompt": {},
        "outputs": {},
        "filename": None,
        "save_node_id": "9",
        "image_bytes": None,
        "_illustration_runtime_snapshot": runtime,
    }

    extracted = iter([
        "(lighting:2.0)\n[ILXL]\nactual prompt after transform\n[UPSCALE]\nfinal tags",
        "low quality",
    ])

    def extract(_prompt, _title):
        return next(extracted)

    async def generate(positive, negative, **_kwargs):
        captured["generated_prompt"] = (positive, negative)
        return raw_pixels, {}

    async def save_backup(image_bytes, _prompt_id, positive, negative, **_kwargs):
        captured["backup_input"] = (image_bytes, positive, negative)
        return "new-backup-name", captioned_pixels

    monkeypatch.setattr(server, "extract_prompts_by_title", extract)
    monkeypatch.setattr(server, "generate_image_with_prompt", generate)
    monkeypatch.setattr(server, "save_backup", save_backup)
    monkeypatch.setattr(server, "log_to_file", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "notify_frontend", lambda *_args, **_kwargs: asyncio.sleep(0))

    try:
        await server.process_prompt(
            prompt_id,
            {},
            {
                "illustration_provider": "comfy",
                "illustration_context_session_id": "session-raw-source",
            },
        )
        entry = server.prompts[prompt_id]
        assert captured["backup_input"] == (
            raw_pixels,
            "(lighting:1.2)\n[ILXL]\nactual prompt after transform\n[UPSCALE]\nfinal tags",
            "low quality",
        )
        assert captured["generated_prompt"] == captured["backup_input"][1:]
        assert "actual prompt after transform" in captured["backup_input"][1]
        assert "(lighting:2.0)" not in captured["backup_input"][1]
        assert entry["image_bytes"] == captioned_pixels
        assert entry["backup_name"] == "new-backup-name"
        assert entry["_quality_inspection_source"] == {
            "image_bytes": raw_pixels,
            "positive": captured["backup_input"][1],
            "negative": captured["backup_input"][2],
        }
        assert entry["_quality_inspection_source"]["image_bytes"] != entry["image_bytes"]
    finally:
        server.prompts.pop(prompt_id, None)


@pytest.mark.asyncio
async def test_context_pipeline_publishes_ready_before_inspection_and_filters_retries_assets_and_failures(
    tmp_path, monkeypatch
):
    session_id = "risu_" + ("f" * 64)
    original_prompt_id = "quality-pipeline-original"
    monkeypatch.setattr(pipeline, "SESSION_DIR", str(tmp_path / "sessions"))
    pipeline.create_session(session_id, "")
    server.prompts[original_prompt_id] = {
        "status": "running",
        "prompt": {},
        "outputs": {},
        "filename": None,
        "save_node_id": "9",
        "image_bytes": None,
    }
    descriptors = [
        {
            "kind": "scene",
            "slot": 7,
            "raw_positive": "scene seven positive",
            "raw_negative": "scene seven negative",
            "scene": "Ari crosses the river.",
        },
        {
            "kind": "scene",
            "slot": 2,
            "raw_positive": "scene two positive",
            "raw_negative": "scene two negative",
            "scene": "Ari finds the lantern.",
        },
        {
            "kind": "scene",
            "slot": 11,
            "raw_positive": "scene eleven positive",
            "raw_negative": "scene eleven negative",
            "scene": "Ari leaves the river.",
        },
    ]
    attempts = {slot: 0 for slot in (2, 7, 11)}
    enqueued_slots: list[int] = []
    captured_inspection: dict = {}
    lifecycle: list[str] = []
    child_prompt_ids: list[str] = []

    async def fake_add_item(item_type, _label, params, **_kwargs):
        assert item_type == "illustration"
        index = int(params["raw_body"]["illustration_context_index"]) - 1
        slot = int(descriptors[index]["slot"])
        attempts[slot] += 1
        enqueued_slots.append(slot)
        child_id = params["prompt_id"]
        child_prompt_ids.append(child_id)
        future = asyncio.get_running_loop().create_future()
        child = SimpleNamespace(id=f"child-{slot}-{attempts[slot]}", status="completed", completion_future=future)
        prompt = server.prompts[child_id]
        if slot == 7 and attempts[slot] == 1:
            child.status = "failed"
            future.set_exception(RuntimeError("first render failed"))
        elif slot == 11:
            child.status = "failed"
            future.set_exception(RuntimeError("terminal render failed"))
        else:
            image = f"raw-image-slot-{slot}-attempt-{attempts[slot]}".encode()
            prompt["image_bytes"] = image
            prompt["backup_name"] = f"new-backup-slot-{slot}"
            prompt["_quality_inspection_source"] = {
                "image_bytes": image,
                "positive": f"actual generated positive {slot}",
                "negative": f"actual generated negative {slot}",
            }
            future.set_result({"success": True})
        return child

    async def fake_asset_selection(**_kwargs):
        return {
            "items": [{"kind": "original_asset", "slot": 99, "backup_name": "uploaded-99"}],
            "images": [b"uploaded-original-asset"],
            "failures": [],
            "requested_count": 1,
        }

    async def fake_inspection(*, session_id, context, entries):
        lifecycle.append("inspection-enqueue")
        captured_inspection.update(session_id=session_id, context=context, entries=copy.deepcopy(entries))

    original_set_result = pipeline.set_session_result

    def record_set_result(*args, **kwargs):
        lifecycle.append("ready-publish")
        return original_set_result(*args, **kwargs)

    async def fake_complete(prompt_id, _save_node_id, _filename):
        server.prompts[prompt_id]["status"] = "completed"

    monkeypatch.setattr(server, "_capture_illustration_runtime_snapshot", lambda: _runtime_snapshot(original_assets=True))
    monkeypatch.setattr(server, "_require_active_illustration_bot", lambda bot_name, *, context: bot_name)
    monkeypatch.setattr(server.queue_manager, "add_item", fake_add_item)
    monkeypatch.setattr(server, "_select_original_asset_outputs", fake_asset_selection)
    monkeypatch.setattr(server, "_enqueue_illustration_quality_inspection", fake_inspection)
    monkeypatch.setattr(server, "complete_prompt_from_reschedule", fake_complete)
    monkeypatch.setattr(server.queue_manager, "_notify_progress", lambda *_args, **_kwargs: asyncio.sleep(0))
    monkeypatch.setattr(server, "set_prompt_by_title", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(pipeline, "set_session_result", record_set_result)

    parent_item = SimpleNamespace(
        id="quality-parent",
        params={
            "prompt_id": original_prompt_id,
            "payload": {
                "protocol": "prompt_batch_v1",
                "session_id": session_id,
                "context": "CALL1 translated context that must not replace the source.",
                "chats": [
                    {"role": "user", "data": "The source says Ari carries a blue coat."},
                    {"role": "char", "data": "Ari crosses the river and keeps the blue coat."},
                ],
                "items": descriptors,
            },
            "prompt_data": {},
            "raw_body": {},
        },
    )

    try:
        result = await server.process_illustration_context_queue_item(parent_item)
        session = pipeline.get_session(session_id)

        assert enqueued_slots == [7, 2, 11, 7, 11]
        assert result == {
            "success": True,
            "session_id": session_id,
            "count": 3,
            "requested_count": 4,
            "failed_count": 1,
        }
        assert [item["slot"] for item in session["items"]] == [2, 7, 99]
        assert session["images"] == [
            b"raw-image-slot-2-attempt-1",
            b"raw-image-slot-7-attempt-2",
            b"uploaded-original-asset",
        ]
        assert lifecycle.index("ready-publish") < lifecycle.index("inspection-enqueue")
        assert captured_inspection["session_id"] == session_id
        assert "Ari crosses the river" in captured_inspection["context"]
        assert "CALL1 translated context" not in captured_inspection["context"]
        assert [entry["slot"] for entry in captured_inspection["entries"]] == [2, 7]
        assert [entry["backup_name"] for entry in captured_inspection["entries"]] == [
            "new-backup-slot-2",
            "new-backup-slot-7",
        ]
        assert [entry["image_bytes"] for entry in captured_inspection["entries"]] == [
            b"raw-image-slot-2-attempt-1",
            b"raw-image-slot-7-attempt-2",
        ]
        assert all("actual generated positive" in entry["positive"] for entry in captured_inspection["entries"])
        assert 11 not in [entry["slot"] for entry in captured_inspection["entries"]]
        assert 99 not in [entry["slot"] for entry in captured_inspection["entries"]]
        for child_id in child_prompt_ids:
            assert "_quality_inspection_source" not in server.prompts[child_id]
    finally:
        pipeline._SESSIONS.pop(session_id, None)
        pipeline._LOOKUP_KEYS.pop("f" * 24, None)
        server.prompts.pop(original_prompt_id, None)
        for child_id in child_prompt_ids:
            server.prompts.pop(child_id, None)


@pytest.mark.asyncio
async def test_quality_history_record_is_text_only_and_shared_across_all_images(monkeypatch):
    records: list[dict] = []

    async def fake_call(_task_key, _messages, **kwargs):
        kwargs["metadata_sink"].update(completion_tokens=12, prompt_tokens=34)
        raw = json.dumps({
            "images": [
                {"slot": 2, "feedback": "The reaching action reads clearly."},
                {"slot": 5, "feedback": "The lantern action is visible."},
            ],
            "overall_feedback": "Ari and the blue coat remain consistent across the set.",
        })
        return SimpleNamespace(
            accepted=True,
            raw_response=raw,
            text=raw,
            final_slot="llm2",
            final_phase="fallback",
            reason="",
            exception=None,
        )

    monkeypatch.setattr(quality.llm_service, "get_config", _inspection_config)
    monkeypatch.setattr(quality.llm_service, "callLLMVisionTaskResult", fake_call)
    monkeypatch.setattr(quality.lighbd_service, "_log_lighbd_history", records.append)

    result = await quality.execute_inspection(
        SimpleNamespace(id="quality-log-queue"),
        session_id="quality-log-session",
        flow_run_id="flow-log",
        context="Ari crosses the river and opens the lantern.",
        entries=_entries(),
    )

    assert result["overall_feedback"].startswith("Ari and the blue coat")
    assert len(records) == 1
    record = records[0]
    assert record["task_key"] == quality.TASK_KEY
    assert [image["slot"] for image in record["inspection_images"]] == [2, 5]
    assert record["inspection_images"] == [
        {"slot": 2, "backup_name": "river-scene-2", "prompt_id": "prompt-2"},
        {"slot": 5, "backup_name": "river-scene-5", "prompt_id": "prompt-5"},
    ]
    record_text = json.dumps(record, ensure_ascii=False)
    assert "Ari and the blue coat remain consistent" in record_text
    assert base64.b64encode(_entries()[0]["image_bytes"]).decode("ascii") not in record_text
    assert "raw-two" not in record_text
    assert "raw-five" not in record_text


def test_quality_history_is_counted_against_general_300_record_cap(tmp_path, monkeypatch):
    history_path = tmp_path / "lighbd_history.jsonl"
    old_records = [
        {
            "history_id": f"old-{index}",
            "task_key": "ordinary_llm_task",
            "ts": f"2026-09-10T00:00:{index:02d}",
            "output": "old",
        }
        for index in range(300)
    ]
    history_path.write_text(
        "".join(json.dumps(record) + "\n" for record in old_records),
        encoding="utf-8",
    )
    monkeypatch.setattr(lighbd_service, "LIGHBD_HISTORY_PATH", str(history_path))
    monkeypatch.setattr(lighbd_service, "LOG_DIR", str(tmp_path / "logs"))

    lighbd_service._log_lighbd_history({
        "history_id": "quality-cap",
        "task_key": quality.TASK_KEY,
        "inspection_images": [{"slot": 2, "backup_name": "cap-image"}],
        "output": "quality feedback",
    })

    records = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == lighbd_service.LIGHBD_GENERAL_HISTORY_MAX
    assert records[-1]["history_id"] == "quality-cap"
    assert records[-1]["task_key"] == quality.TASK_KEY


def test_llm_trace_lookup_uses_exact_inspection_backup_name_even_with_empty_metadata_trace(
    tmp_path, monkeypatch
):
    history_path = tmp_path / "lighbd_history.jsonl"
    exact = {
        "ts": "2026-09-10T12:00:01",
        "history_id": "inspection-exact",
        "task_key": quality.TASK_KEY,
        "inspection_images": [{"slot": 2, "backup_name": "regenerated-scene-2"}],
    }
    prefix_collision = {
        "ts": "2026-09-10T12:00:02",
        "history_id": "inspection-newer",
        "task_key": quality.TASK_KEY,
        "inspection_images": [{"slot": 2, "backup_name": "regenerated-scene-2-new"}],
    }
    unrelated = {
        "ts": "2026-09-10T12:00:03",
        "history_id": "ordinary-match",
        "task_key": "ordinary_llm_task",
        "inspection_images": [{"slot": 2, "backup_name": "regenerated-scene-2"}],
    }
    history_path.write_text(
        "".join(json.dumps(record) + "\n" for record in (exact, prefix_collision, unrelated)),
        encoding="utf-8",
    )
    monkeypatch.setattr(server.lighbd_service, "LIGHBD_HISTORY_PATH", str(history_path))

    trace_ids, records, missing = server._load_llm_trace_records(
        [],
        log_prefix="TEST",
        context="empty metadata trace",
        inspection_backup_name="regenerated-scene-2",
    )

    assert trace_ids == ["inspection-exact"]
    assert [record["history_id"] for record in records] == ["inspection-exact"]
    assert missing == []


@pytest.mark.asyncio
async def test_backup_llm_trace_api_returns_inspection_when_info_has_no_trace_ids(tmp_path, monkeypatch):
    name = "regenerated-scene-2"
    (tmp_path / f"{name}_info.json").write_text(
        json.dumps({"bot_name": "", "llm_trace": []}),
        encoding="utf-8",
    )
    history_path = tmp_path / "lighbd_history.jsonl"
    history_path.write_text(
        "\n".join(
            json.dumps(record)
            for record in (
                {
                    "ts": "2026-09-10T12:00:01",
                    "history_id": "inspection-api-exact",
                    "task_key": quality.TASK_KEY,
                    "inspection_images": [
                        {"slot": 2, "backup_name": name},
                        {"slot": 5, "backup_name": "other-scene"},
                    ],
                },
                {
                    "ts": "2026-09-10T12:00:02",
                    "history_id": "inspection-api-prefix",
                    "task_key": quality.TASK_KEY,
                    "inspection_images": [{"slot": 2, "backup_name": f"{name}-new"}],
                },
            )
        ) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(server, "get_backup_base_dir", lambda: str(tmp_path))
    monkeypatch.setattr(server.lighbd_service, "LIGHBD_HISTORY_PATH", str(history_path))

    response = await server.handle_api_backup_llm_trace(_Request("GET", name=name))
    payload = json.loads(response.text)

    assert response.status == 200
    assert payload["trace_ids"] == ["inspection-api-exact"]
    assert [record["history_id"] for record in payload["records"]] == ["inspection-api-exact"]
    assert payload["missing"] == []


@pytest.mark.asyncio
async def test_quality_background_node_shares_illustration_run_without_blocking_image_result(monkeypatch):
    monkeypatch.setattr(illustration_flow, "_latest", None)
    root = SimpleNamespace(
        id="root-illustration",
        type="illustration_llm_build",
        label="CALL1/2/3",
        params={},
        status="pending",
    )
    illustration_flow.queue_added(root)
    run = illustration_flow._latest
    assert run is not None

    token_run = illustration_flow._run.set(run)
    token_frontier = illustration_flow._frontier.set((root.id,))
    try:
        background = SimpleNamespace(
            id="background-inspection",
            type=quality.TASK_KEY,
            label="quality inspection",
            params={"session_id": "session-flow"},
            status="pending",
        )
        illustration_flow.queue_added(background)
        image = SimpleNamespace(
            id="image-result",
            type="illustration",
            label="slot 2",
            params={"provider": "comfy"},
            status="pending",
        )
        illustration_flow.queue_added(image)
    finally:
        illustration_flow._frontier.reset(token_frontier)
        illustration_flow._run.reset(token_run)

    await asyncio.sleep(0)

    graph = illustration_flow.snapshot(run)
    nodes = {node["id"]: node for node in graph["nodes"]}
    assert background._illustration_flow[0] is run
    assert nodes[background.id]["kind"] == "llm"
    assert nodes[image.id]["kind"] == "image"
    assert background.id not in nodes[image.id]["dependencies"]
    assert nodes[image.id]["dependencies"] == [root.id]
