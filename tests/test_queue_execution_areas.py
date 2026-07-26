import asyncio

import pytest

from queue_manager import QueueItem, QueueManager


def _item(item_type, params=None):
    return QueueItem(
        id=f"{item_type}-id",
        type=item_type,
        label=item_type,
        params=params or {},
    )


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
    ]

    status = manager.get_status()

    assert [item["execution_area"] for item in status["items"]] == [
        "llm",
        "gpu",
        "external",
    ]
    assert [item["provider"] for item in status["items"]] == [
        "llm",
        "comfy",
        "chansub",
    ]


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
    """회귀 보호: priority >= 10 의존성(refine → training)은 고순위 면제 밖이라
    블로킹 검사가 그대로 유지돼야 한다. 면제가 10 미만에만 적용되는지 확인.
    """
    manager = QueueManager()
    manager.get_config = lambda: {
        "queue_type_order": {
            "instance_lora_analysis": 4,
            "instance_lora_training": 5,
        }
    }

    refine = QueueItem(
        id="refine", type="instance_lora_prompt_refine", label="refine", priority=10
    )
    refine.status = "processing"
    training = QueueItem(
        id="train", type="instance_lora_training", label="train", priority=10
    )
    manager.items = [refine, training]

    executed = await _run_process_loop_with_fake_pipeline(manager)

    assert executed == []
