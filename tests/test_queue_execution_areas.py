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
