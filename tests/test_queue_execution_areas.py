import asyncio

import pytest

from queue_manager import (
    GPU_QUEUE_PRIORITY_TYPES,
    LLM_QUEUE_PRIORITY_TYPES,
    QueueItem,
    QueueManager,
    normalize_queue_priority_orders,
)


def _item(item_type, params=None):
    return QueueItem(
        id=f"{item_type}-id",
        type=item_type,
        label=item_type,
        params=params or {},
    )


@pytest.mark.asyncio
async def test_illustration_llm_build_overlaps_modal_warm_lease_and_releases_it():
    manager = QueueManager()
    manager.get_config = lambda: {
        "modal_enabled": True,
        "modal_max_concurrency": 2,
        "illustration_provider": "comfy",
        "bot_selected": "test-bot",
        "comfy_task_allocations": {"illustration": "modal"},
    }
    acquire_started = asyncio.Event()
    pipeline_started = asyncio.Event()
    events = []

    async def acquire(*, reason):
        events.append(("acquire", reason))
        acquire_started.set()
        await pipeline_started.wait()
        return "warm-token"

    async def release(token, *, reason):
        events.append(("release", token, reason))
        return True

    async def process(item):
        events.append(("pipeline", item.id))
        pipeline_started.set()
        await acquire_started.wait()
        return {"success": True}

    manager.acquire_modal_warm_lease = acquire
    manager.release_modal_warm_lease = release
    manager.process_illustration_context = process
    item = _item("illustration_llm_build")

    result = await manager._handle_illustration_llm_build(item)

    assert result == {"success": True}
    assert {event[0] for event in events[:2]} == {"acquire", "pipeline"}
    assert events[-1] == (
        "release",
        "warm-token",
        f"illustration_llm_build:{item.id}",
    )


@pytest.mark.asyncio
async def test_illustration_llm_build_skips_warm_lease_for_local_only_allocation():
    manager = QueueManager()
    manager.get_config = lambda: {
        "modal_enabled": True,
        "illustration_provider": "comfy",
        "bot_selected": "test-bot",
        "comfy_task_allocations": {"illustration": 1},
        "comfy_task_modal_parallel": {"illustration": False},
    }
    calls = []

    async def acquire(*, reason):
        calls.append(("acquire", reason))
        return "unexpected"

    async def release(token, *, reason):
        calls.append(("release", token, reason))
        return True

    async def process(_item):
        return {"success": True}

    manager.acquire_modal_warm_lease = acquire
    manager.release_modal_warm_lease = release
    manager.process_illustration_context = process

    result = await manager._handle_illustration_llm_build(
        _item("illustration_llm_build")
    )

    assert result == {"success": True}
    assert calls == []


@pytest.mark.asyncio
async def test_illustration_llm_build_releases_warm_lease_after_pipeline_failure():
    manager = QueueManager()
    manager.get_config = lambda: {
        "modal_enabled": True,
        "illustration_provider": "comfy",
        "bot_selected": "test-bot",
        "comfy_task_allocations": {"illustration": "modal"},
    }
    released = []

    async def acquire(*, reason):
        return f"warm-token:{reason}"

    async def release(token, *, reason):
        released.append((token, reason))
        return True

    async def process(_item):
        raise RuntimeError("synthetic illustration pipeline failure")

    manager.acquire_modal_warm_lease = acquire
    manager.release_modal_warm_lease = release
    manager.process_illustration_context = process
    item = _item("illustration_llm_build")

    with pytest.raises(RuntimeError, match="synthetic illustration pipeline failure"):
        await manager._handle_illustration_llm_build(item)

    reason = f"illustration_llm_build:{item.id}"
    assert released == [(f"warm-token:{reason}", reason)]


@pytest.mark.asyncio
async def test_completion_future_failure_is_observed_without_swallowing_await_error():
    manager = QueueManager()
    future = asyncio.get_running_loop().create_future()
    future.add_done_callback(manager._mark_completion_future_observed)

    future.set_exception(RuntimeError("synthetic queue failure"))
    await asyncio.sleep(0)

    assert getattr(future, "_log_traceback", False) is False
    with pytest.raises(RuntimeError, match="synthetic queue failure"):
        await future


def test_queue_status_separates_llm_gpu_and_chansub_areas():
    manager = QueueManager()
    manager.get_config = lambda: {
        "bot_selected": "test-bot",
        "illustration_provider": "comfy",
    }
    manager.items = [
        _item("illustration_llm_build"),
        _item("illustration", {"provider": "comfy"}),
        _item("illustration", {"provider": "chansub"}),
        _item("illustration", {"provider": "hybrid"}),
    ]

    status = manager.get_status()

    assert [item["execution_area"] for item in status["items"]] == [
        "llm",
        "gpu",
        "external",
        "hybrid",
    ]
    assert [item["provider"] for item in status["items"]] == [
        "llm",
        "comfy",
        "chansub",
        "hybrid",
    ]


def test_video_modes_share_one_comfy_allocation_but_keep_detailed_queue_labels():
    manager = QueueManager()

    for item_type in ("video_i2v", "video_first_last"):
        assert manager._comfy_task_key_for_item(_item(item_type)) == "video_generation"
        assert manager._item_execution_area(_item(item_type))[0] == "gpu"

    assert manager._item_execution_area(_item("video_prompt_build"))[0] == "llm"
    assert manager._item_execution_area(_item("video_instruction_draft"))[0] == "llm"
    assert manager._item_execution_area(_item("video_postprocess")) == (
        "video_postprocess",
        "realesrgan-ncnn-vulkan",
    )


@pytest.mark.asyncio
async def test_video_postprocess_worker_overlaps_local_comfy_lane(monkeypatch):
    manager = QueueManager()
    manager.get_config = lambda: {"llm_max_concurrency": 1}
    both_started = asyncio.Event()
    release = asyncio.Event()
    started_areas = set()

    async def fake_execute(item):
        started_areas.add(manager._item_execution_area(item)[0])
        if started_areas == {"gpu", "video_postprocess"}:
            both_started.set()
        await release.wait()
        return {"success": True}

    async def no_prune(_item):
        return None

    monkeypatch.setattr(manager, "_execute_item", fake_execute)
    monkeypatch.setattr(manager, "_deferred_prune", no_prune)

    gpu_item = await manager.add_item("asset_generation", "asset", {})
    post_item = await manager.add_item(
        "video_postprocess",
        "video postprocess",
        {"job_dir": "synthetic"},
    )

    try:
        await asyncio.wait_for(both_started.wait(), timeout=1)
        assert gpu_item.status == "processing"
        assert post_item.status == "processing"
        assert manager.get_status()["video_postprocess_active"] == 1

        release.set()
        await asyncio.wait_for(
            asyncio.gather(gpu_item.completion_future, post_item.completion_future),
            timeout=1,
        )
    finally:
        release.set()
        tasks = [
            task
            for task in (
                manager._video_postprocess_worker_task,
                *manager._llm_worker_tasks.values(),
            )
            if task is not None and not task.done()
        ]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_video_llm_stage_enqueues_matching_gpu_type_only_after_success():
    manager = QueueManager()
    calls = []

    class FakeVideoMode:
        async def build_prompt(self, params, queue_item_id=""):
            calls.append(("build", params["mode"], queue_item_id))
            return {
                "success": True,
                "h3_prompt": "synthetic H3 prompt",
                "llm_trace": ["trace-1"],
                "history_id": "trace-1",
            }

    async def fake_add_item(item_type, label, params, **_kwargs):
        calls.append(("enqueue", item_type, label, params))
        return _item(item_type)

    manager.video_mode = FakeVideoMode()
    manager.add_item = fake_add_item
    item = _item(
        "video_prompt_build",
        {"mode": "first_last", "source_backup": "first", "last_backup": "last"},
    )

    result = await manager._handle_video_prompt_build(item)

    assert calls[0] == ("build", "first_last", item.id)
    assert calls[1][0:2] == ("enqueue", "video_first_last")
    assert calls[1][3]["h3_prompt"] == "synthetic H3 prompt"
    assert result["render_item_id"] == "video_first_last-id"


@pytest.mark.asyncio
async def test_video_instruction_draft_returns_text_without_enqueuing_gpu_work():
    manager = QueueManager()
    calls = []

    class FakeVideoMode:
        async def build_instruction_draft(self, params, queue_item_id=""):
            calls.append(("draft", params["mode"], queue_item_id))
            return {
                "success": True,
                "draft": "인물이 천천히 미소 짓는다.",
                "language": "ko",
            }

    async def reject_add_item(*_args, **_kwargs):
        raise AssertionError("draft handler must not enqueue GPU work")

    manager.video_mode = FakeVideoMode()
    manager.add_item = reject_add_item
    item = _item("video_instruction_draft", {"mode": "i2v"})

    result = await manager._handle_video_instruction_draft(item)

    assert calls == [("draft", "i2v", item.id)]
    assert result["draft"] == "인물이 천천히 미소 짓는다."


def test_pending_hybrid_item_moves_to_claimed_queue_area():
    manager = QueueManager()
    item = _item("illustration", {
        "provider": "hybrid",
        "raw_body": {"illustration_provider": "hybrid"},
        "hybrid_prompt_formats": {"comfy": "v3", "chansub": "chansub"},
    })
    manager.items = [item]

    assert manager.get_status()["items"][0]["execution_area"] == "hybrid"

    assert manager._bind_hybrid_item_provider(item, "chansub") is True
    moved = manager.get_status()["items"][0]
    assert moved["execution_area"] == "external"
    assert moved["provider"] == "chansub"
    assert item.params["raw_body"]["illustration_provider_mode"] == "hybrid"
    assert item.params["raw_body"]["illustration_prompt_format"] == "chansub"


def test_queue_status_forces_plain_illustration_to_gpu_without_active_bot():
    manager = QueueManager()
    manager.get_config = lambda: {
        "bot_selected": "",
        "illustration_provider": "chansub",
    }
    manager.items = [_item("illustration")]

    item = manager.get_status()["items"][0]

    assert item["execution_area"] == "gpu"
    assert item["provider"] == "comfy"


def test_chansub_external_worker_default_and_bounds():
    manager = QueueManager()
    manager.get_config = lambda: {}
    assert manager._target_external_workers() == 1

    manager.get_config = lambda: {"chansub_max_concurrency": 2}
    assert manager._target_external_workers() == 2

    manager.get_config = lambda: {"chansub_max_concurrency": 99}
    assert manager._target_external_workers() == 2

    manager.get_config = lambda: {"chansub_max_concurrency": 1.5}
    assert manager._target_external_workers() == 1


@pytest.mark.asyncio
async def test_two_chansub_workers_execute_two_requests_concurrently(monkeypatch):
    manager = QueueManager()
    manager.get_config = lambda: {
        "bot_selected": "test-bot",
        "illustration_provider": "chansub",
        "chansub_max_concurrency": 2,
    }
    both_started = asyncio.Event()
    release = asyncio.Event()
    started = []

    async def fake_execute(item):
        started.append(item.id)
        if len(started) == 2:
            both_started.set()
        await release.wait()
        return {"success": True}

    async def no_llm_workers():
        return None

    async def no_prune(_item):
        return None

    monkeypatch.setattr(manager, "_execute_item", fake_execute)
    monkeypatch.setattr(manager, "_ensure_llm_workers", no_llm_workers)
    monkeypatch.setattr(manager, "_deferred_prune", no_prune)

    first = await manager.add_item(
        "illustration",
        "remote-1",
        {"provider": "chansub"},
        priority=0,
    )
    second = await manager.add_item(
        "illustration",
        "remote-2",
        {"provider": "chansub"},
        priority=0,
    )

    try:
        await asyncio.wait_for(both_started.wait(), timeout=1)
        status = manager.get_status()
        assert first.status == second.status == "processing"
        assert {item["id"] for item in status["current_externals"]} == {
            first.id,
            second.id,
        }
        assert status["external_active_workers"] == 2
        assert status["external_target_workers"] == 2

        release.set()
        await asyncio.wait_for(
            asyncio.gather(first.completion_future, second.completion_future),
            timeout=1,
        )
        assert first.status == second.status == "completed"
    finally:
        release.set()
        tasks = [
            task
            for task in manager._external_worker_tasks.values()
            if not task.done()
        ]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.parametrize("fast_provider", ["comfy", "chansub"])
@pytest.mark.asyncio
async def test_faster_hybrid_lane_claims_remaining_pending_items(
    monkeypatch,
    fast_provider,
):
    manager = QueueManager()
    manager.get_config = lambda: {
        "bot_selected": "test-bot",
        "illustration_provider": "hybrid",
        "chansub_max_concurrency": 1,
    }
    initial_lanes_started = asyncio.Event()
    all_items_started = asyncio.Event()
    release_gpu = asyncio.Event()
    release_external = asyncio.Event()
    starts = []

    async def fake_execute(item):
        provider = item.params["provider"]
        starts.append((item.id, provider))
        if len(starts) >= 2:
            initial_lanes_started.set()
        if len(starts) == 4:
            all_items_started.set()
        if provider == "comfy":
            await release_gpu.wait()
        else:
            await release_external.wait()
        return {"success": True, "provider": provider}

    async def no_llm_workers():
        return None

    async def no_wait():
        return None

    async def no_prune(_item):
        return None

    monkeypatch.setattr(manager, "_execute_item", fake_execute)
    monkeypatch.setattr(manager, "_ensure_llm_workers", no_llm_workers)
    monkeypatch.setattr(manager, "_wait_after_illustration", no_wait)
    monkeypatch.setattr(manager, "_deferred_prune", no_prune)

    items = []
    for index in range(4):
        items.append(await manager.add_item(
            "illustration",
            f"dynamic-{index}",
            {
                "provider": "hybrid",
                "raw_body": {"illustration_provider": "hybrid"},
                "hybrid_prompt_formats": {
                    "comfy": "v3",
                    "chansub": "chansub",
                },
            },
            priority=0,
        ))

    try:
        await asyncio.wait_for(initial_lanes_started.wait(), timeout=1)
        assert {provider for _, provider in starts[:2]} == {"comfy", "chansub"}

        # 한 레인만 먼저 풀면 남은 두 작업도 먼저 빈 같은 레인이 연속으로 가져간다.
        if fast_provider == "comfy":
            release_gpu.set()
        else:
            release_external.set()
        await asyncio.wait_for(all_items_started.wait(), timeout=1)
        slow_provider = "chansub" if fast_provider == "comfy" else "comfy"
        assert [provider for _, provider in starts].count(fast_provider) == 3
        assert [provider for _, provider in starts].count(slow_provider) == 1

        release_external.set()
        release_gpu.set()
        await asyncio.wait_for(
            asyncio.gather(*(item.completion_future for item in items)),
            timeout=1,
        )
        assert all(item.status == "completed" for item in items)
    finally:
        release_external.set()
        release_gpu.set()
        tasks = [
            task
            for task in manager._external_worker_tasks.values()
            if not task.done()
        ]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_queue_subtasks_are_prepopulated_and_updated_on_parent_item():
    manager = QueueManager()
    notifications = []

    async def notify(event_type, data):
        notifications.append((event_type, data))

    manager.notify_frontend = notify
    item = _item("illustration_llm_build")
    manager.items = [item]
    metadata = {
        "group_id": "backtranslation",
        "group_label": "역번역",
        "index": 2,
        "total": 3,
    }

    updated = await manager.update_subtask(item, metadata, {"type": "start"})

    assert updated is True
    assert [subtask["status"] for subtask in item.subtasks] == [
        "pending",
        "processing",
        "pending",
    ]
    assert [subtask["label"] for subtask in item.subtasks] == [
        "역번역 1/3",
        "역번역 2/3",
        "역번역 3/3",
    ]
    assert manager.get_status()["items"][0]["subtasks"] == item.subtasks
    assert notifications[-1][0] == "queue_progress"
    assert notifications[-1][1]["subtasks"][1]["status"] == "processing"

    await manager.update_subtask(item, metadata, {"type": "done"})

    assert item.subtasks[1]["status"] == "completed"
    assert item.subtasks[1]["completed_at"] is not None


@pytest.mark.asyncio
async def test_queue_subtask_failure_keeps_error_on_child():
    manager = QueueManager()
    item = _item("illustration_llm_build")
    metadata = {
        "group_id": "backtranslation",
        "group_label": "역번역",
        "index": 1,
        "total": 1,
    }

    await manager.update_subtask(
        item,
        metadata,
        {"type": "error", "error": "번역 API 실패"},
    )

    assert len(item.subtasks) == 1
    subtask = item.subtasks[0]
    assert subtask["id"] == "backtranslation:1"
    assert subtask["label"] == "역번역 1/1"
    assert subtask["status"] == "failed"
    assert subtask["started_at"] is not None
    assert subtask["completed_at"] is not None
    assert subtask["error"] == "번역 API 실패"


@pytest.mark.asyncio
async def test_multi_char_mask_is_prepared_at_illustration_execution_time(monkeypatch, tmp_path):
    from modes import multi_char_mask

    manager = QueueManager()
    events = []
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    manager.get_config = lambda: {"comfy_input_dir": str(input_dir)}

    def fake_prepare(comfy_input_dir, context, mask_location):
        events.append(("mask", comfy_input_dir, context["enable"], mask_location))
        return str(input_dir / "region_mask" / "region_mask.png")

    async def fake_process(prompt_id, prompt_data, raw_body, queue_progress_callback=None):
        events.append(("process", prompt_id))

    monkeypatch.setattr(multi_char_mask, "prepare_region_mask", fake_prepare)
    manager.process_prompt_full = fake_process
    item = _item("illustration", {
        "prompt_id": "prompt-id",
        "prompt_data": {},
        "raw_body": {
            "illustration_multi_char": {
                "enable": True,
                "mask_location": "region_mask",
            }
        },
    })

    result = await manager._handle_illustration(item)

    assert [event[0] for event in events] == ["mask", "process"]
    assert result == {"success": True, "prompt_id": "prompt-id"}


def _regenerate_multi_char_fixture():
    from modes import multi_char_mask

    layout = {
        "background_prompt": "wide shot, rooftop",
        "composition_prompt": "two distinct people standing apart",
        "regions": [
            {
                "name": "Left",
                "character_prompt": "grey hair, holding a chart",
                "x": 0.0,
                "y": 0.0,
                "width": 0.55,
                "height": 1.0,
            },
            {
                "name": "Right",
                "character_prompt": "black hair, pointing upward",
                "x": 0.45,
                "y": 0.0,
                "width": 0.55,
                "height": 1.0,
            },
        ],
    }
    snapshot = multi_char_mask.normalize_multi_char_snapshot({
        "enable": True,
        "character_order": ["Left", "Right"],
        "layout": layout,
        "mask_location": "region_mask",
    })
    positive = "\n".join([
        "[MULTI_CHAR]",
        (
            '{"enable": true, "char_num": 2, '
            '"char_name_list": ["Left", "Right"], '
            f'"mask_fingerprint": "{snapshot["mask_fingerprint"]}"}}'
        ),
        "[HRF_ACTIVATE]",
        "false",
    ])
    return snapshot, positive


@pytest.mark.asyncio
async def test_multi_char_mask_is_restored_before_regenerate_and_inherited_by_backup(
    monkeypatch,
    tmp_path,
):
    from modes import multi_char_mask

    manager = QueueManager()
    events = []
    saved = {}
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    manager.get_config = lambda: {"comfy_input_dir": str(input_dir)}
    snapshot, positive = _regenerate_multi_char_fixture()

    def fake_prepare(comfy_input_dir, context, mask_location):
        events.append("mask")
        assert context["mask_fingerprint"] == snapshot["mask_fingerprint"]
        return str(input_dir / "region_mask" / "region_mask.png")

    async def fake_generate(*args, **kwargs):
        events.append("generate")
        return b"generated-image", None

    async def fake_save(*args, **kwargs):
        events.append("save")
        saved.update(kwargs)
        return "new-backup", args[0]

    monkeypatch.setattr(multi_char_mask, "prepare_region_mask", fake_prepare)
    manager.generate_image_with_prompt = fake_generate
    manager.save_backup = fake_save
    item = _item("regenerate", {
        "backup_name": "old-backup",
        "positive": positive,
        "negative": "",
        "provider": "comfy",
        "illustration_multi_char": snapshot,
    })

    result = await manager._handle_regenerate(item)

    assert events == ["mask", "generate", "save"]
    assert saved["illustration_multi_char"] == snapshot
    assert result["success"] is True
    assert result["backup_name"] == "new-backup"
    assert item.generated_image_bytes == b"generated-image"


@pytest.mark.asyncio
async def test_multi_char_regenerate_fails_before_generation_without_mask_snapshot():
    manager = QueueManager()
    snapshot, positive = _regenerate_multi_char_fixture()
    generated = False

    async def fake_generate(*args, **kwargs):
        nonlocal generated
        generated = True
        return b"unexpected", None

    manager.generate_image_with_prompt = fake_generate
    item = _item("regenerate", {
        "backup_name": "missing-mask-backup",
        "positive": positive,
        "negative": "",
        "provider": "comfy",
    })

    with pytest.raises(ValueError, match="마스크 스냅샷"):
        await manager._handle_regenerate(item)

    assert snapshot["enable"] is True
    assert generated is False


async def _run_process_loop_with_fake_pipeline(manager):
    """_process_loop를 돌리되 _run_item_pipeline을 즉시 완료 처리하는 stub로 교체.
    어떤 GPU 항목이 실제로 실행 차례가 됐는지 executed 리스트로 관찰한다."""
    executed = []

    async def fake_run(item, is_gpu):
        item.status = "processing"
        executed.append(item.id)
        item.status = "completed"

    manager._run_item_pipeline = fake_run
    await manager._process_loop()
    return executed


@pytest.mark.asyncio
async def test_multi_char_illustration_not_blocked_by_parent_llm_build():
    """회귀: illustration_llm_build(priority 0, processing)가 다중 캐릭터 삽화
    (priority 1, pending)를 블록해 교착에 빠지지 않아야 한다.

    부모 llm_build가 자식 multi-char 완료를 await 중인데 multi-char이 부모 때문에
    실행되지 못하면 다중 캐릭터 큐가 시작 직전에 멈추는 교착이 발생한다.
    priority < 10 고순위 티어 상호 면제로 이 순환 대기를 끊어야 한다.
    """
    manager = QueueManager()
    manager.get_config = lambda: {}

    llm_build = QueueItem(
        id="build", type="illustration_llm_build", label="build", priority=0
    )
    llm_build.status = "processing"
    multi = QueueItem(id="multi", type="illustration", label="multi", priority=1)
    manager.items = [llm_build, multi]

    executed = await _run_process_loop_with_fake_pipeline(manager)

    assert executed == ["multi"]


@pytest.mark.asyncio
async def test_refine_still_blocks_training_under_priority_ten():
    """같은 인스턴스의 LLM 정제가 끝나기 전에는 GPU 학습을 시작하지 않는다."""
    manager = QueueManager()
    manager.get_config = lambda: {
        "queue_type_order": {
            "instance_lora_analysis": 14,
            "instance_lora_training": 15,
        },
        "llm_queue_type_order": {
            "instance_lora_prompt_refine": 11,
        },
    }

    refine = QueueItem(
        id="refine",
        type="instance_lora_prompt_refine",
        label="refine",
        priority=10,
        params={"source_type": "instance", "lora_id": "target"},
    )
    refine.status = "processing"
    training = QueueItem(
        id="train",
        type="instance_lora_training",
        label="train",
        priority=10,
        params={"id": "target", "profiles": ["anima"]},
    )
    manager.items = [refine, training]

    executed = await _run_process_loop_with_fake_pipeline(manager)

    assert executed == []


@pytest.mark.asyncio
async def test_unrelated_llm_work_does_not_block_gpu_lane():
    manager = QueueManager()
    manager.get_config = lambda: {
        "queue_type_order": {"asset_generation": 10},
        "llm_queue_type_order": {"character_maker": 10},
    }

    character_maker = QueueItem(
        id="character",
        type="character_maker",
        label="character",
        priority=10,
        params={"session_id": "session", "payload": {}},
    )
    character_maker.status = "processing"
    gpu_item = QueueItem(
        id="asset",
        type="asset_generation",
        label="asset",
        priority=10,
    )
    manager.items = [character_maker, gpu_item]

    executed = await _run_process_loop_with_fake_pipeline(manager)

    assert executed == ["asset"]


@pytest.mark.asyncio
async def test_gpu_and_llm_workers_execute_unrelated_items_concurrently(monkeypatch):
    manager = QueueManager()
    manager.get_config = lambda: {
        "queue_type_order": {"asset_generation": 10},
        "llm_queue_type_order": {"character_maker": 10},
        "llm_max_concurrency": 1,
    }
    both_started = asyncio.Event()
    release = asyncio.Event()
    started_areas = set()

    async def fake_execute(item):
        started_areas.add(manager._item_execution_area(item)[0])
        if started_areas == {"gpu", "llm"}:
            both_started.set()
        await release.wait()
        return {"success": True}

    async def no_prune(_item):
        return None

    monkeypatch.setattr(manager, "_execute_item", fake_execute)
    monkeypatch.setattr(manager, "_deferred_prune", no_prune)

    gpu_item = await manager.add_item("asset_generation", "asset", {})
    llm_item = await manager.add_item(
        "character_maker",
        "character",
        {"session_id": "session", "payload": {}},
    )

    try:
        await asyncio.wait_for(both_started.wait(), timeout=1)
        assert gpu_item.status == "processing"
        # LLM 항목은 게이트 획득 전까지 'waiting'이다. fake_execute 가 게이트를
        # 호출하지 않으므로 worker 가 잡고 실행 중(in-flight)인 상태로 'waiting'에 머문다.
        # 두 레인이 동시에 in-flight(gpu=processing, llm=waiting)인 것이 병렬 실행의 증거.
        assert llm_item.status == "waiting"

        release.set()
        await asyncio.wait_for(
            asyncio.gather(
                gpu_item.completion_future,
                llm_item.completion_future,
            ),
            timeout=1,
        )
    finally:
        release.set()
        worker_tasks = [
            task
            for task in manager._llm_worker_tasks.values()
            if not task.done()
        ]
        for task in worker_tasks:
            task.cancel()
        if worker_tasks:
            await asyncio.gather(*worker_tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_unrelated_refine_does_not_block_other_instance_training():
    manager = QueueManager()
    manager.get_config = lambda: {}
    refine = QueueItem(
        id="refine-a",
        type="instance_lora_prompt_refine",
        label="refine-a",
        params={"source_type": "instance", "lora_id": "a"},
    )
    refine.status = "processing"
    training = QueueItem(
        id="train-b",
        type="instance_lora_training",
        label="train-b",
        params={"id": "b", "profiles": ["anima"]},
    )
    manager.items = [refine, training]

    executed = await _run_process_loop_with_fake_pipeline(manager)

    assert executed == ["train-b"]


def test_llm_item_waits_for_explicit_gpu_dependency():
    manager = QueueManager()
    analysis = QueueItem(
        id="analysis",
        type="instance_lora_analysis",
        label="analysis",
        params={"lora_id": "target"},
    )
    refine = QueueItem(
        id="refine",
        type="instance_lora_prompt_refine",
        label="refine",
        params={"source_type": "instance", "lora_id": "target"},
        depends_on=["analysis"],
    )
    manager.items = [analysis, refine]

    assert manager._pop_next_llm_item() is None

    analysis.status = "completed"
    assert manager._pop_next_llm_item() is refine
    # LLM 항목은 게이트 획득 전까지 'waiting', _run_item_pipeline 안에서 실제 API
    # 호출(게이트 획득) 시 'processing'으로 전환된다.
    assert refine.status == "waiting"


def test_queue_priority_normalization_registers_every_non_illustration_type():
    gpu_order, llm_order = normalize_queue_priority_orders({
        "queue_type_order": {
            "tag_analysis": 1,
            "asset_generation": 2,
            "bot_llm_face_tag_analysis": 3,
        },
    })

    assert set(gpu_order) == set(GPU_QUEUE_PRIORITY_TYPES)
    assert set(llm_order) == set(LLM_QUEUE_PRIORITY_TYPES)
    assert list(gpu_order.values()) == list(
        range(10, 10 + len(GPU_QUEUE_PRIORITY_TYPES))
    )
    assert list(llm_order.values()) == list(
        range(10, 10 + len(LLM_QUEUE_PRIORITY_TYPES))
    )
    assert llm_order["bot_llm_face_tag_analysis"] == 10


@pytest.mark.asyncio
async def test_character_maker_item_lands_in_llm_lane():
    manager = QueueManager()
    manager.get_config = lambda: {}
    manager.items = [_item("character_maker", {"session_id": "s1", "payload": {}})]

    assert manager.get_status()["items"][0]["execution_area"] == "llm"


@pytest.mark.asyncio
async def test_character_maker_illustration_uses_selected_provider_lane_and_keeps_bytes_off_result():
    manager = QueueManager()
    observed = {}

    async def fake_generate(positive, negative, **kwargs):
        observed.update({"positive": positive, "negative": negative, **kwargs})
        return b"generated-image", None

    manager.generate_image_with_prompt = fake_generate
    item = _item(
        "character_maker_illustration",
        {
            "positive": "[ANIMA_ALL]\nsilver_hair\n[END]",
            "negative": "low quality",
            "provider": "chansub",
            "illustration_workflow_type": "chansub",
            "width": 700,
            "height": 1024,
        },
    )

    assert manager._item_execution_area(item) == ("external", "chansub")
    result = await manager._handle_character_maker_illustration(item)

    assert result["success"] is True
    assert result["image_size"] == len(b"generated-image")
    assert item.generated_image_bytes == b"generated-image"
    assert "generated_image_bytes" not in result
    assert observed["provider"] == "chansub"
    assert observed["illustration_workflow_type"] == "chansub"


@pytest.mark.asyncio
async def test_character_maker_handler_calls_revise_and_returns_result():
    manager = QueueManager()
    received = {}

    class FakeCM:
        async def revise(self, session_id, payload):
            received["session_id"] = session_id
            received["payload"] = payload
            return {"ok": True, "session_id": session_id}

    manager.character_maker = FakeCM()
    item = _item("character_maker", {"session_id": "s1", "payload": {"message": "hi"}})

    result = await manager._execute_item(item)

    assert result == {"ok": True, "session_id": "s1"}
    assert received == {"session_id": "s1", "payload": {"message": "hi"}}
    assert item.progress == 100.0


@pytest.mark.asyncio
async def test_character_maker_handler_raises_when_revise_raises():
    manager = QueueManager()

    class FakeCM:
        async def revise(self, session_id, payload):
            raise RuntimeError("LLM 호출 실패")

    manager.character_maker = FakeCM()
    item = _item("character_maker", {"session_id": "s1", "payload": {}})

    with pytest.raises(RuntimeError, match="LLM 호출 실패"):
        await manager._handle_character_maker(item)


@pytest.mark.asyncio
async def test_character_maker_handler_errors_when_instance_not_injected():
    manager = QueueManager()
    item = _item("character_maker", {"session_id": "s1", "payload": {}})

    with pytest.raises(RuntimeError, match="character_maker 인스턴스가 큐에 주입되지 않았습니다"):
        await manager._handle_character_maker(item)
