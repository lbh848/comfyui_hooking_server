import asyncio
import importlib
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest


def _card(card_id, label, filename, *, aliases=None, guide=""):
    return {
        "id": card_id,
        "label": label,
        "selection_guide": guide,
        "aliases": list(aliases or []),
        "appearance": [{"tag": "blue hair"}],
        "default_outfit": [{"tag": "school uniform"}],
        "rep_images": [filename],
        "use_profile_embedding": card_id != "card_1",
    }


def _bot_data(cards):
    return {
        "bots": [
            {
                "name": "demo",
                "system_prompt_preset": "기본",
                "preset_scope": "local",
                "characters": [{"name": "Riko", "visual_cards": deepcopy(cards)}],
            }
        ],
        "system_prompt_presets": {
            "기본": (
                "Arbitrary Picture Grammar\n"
                "Riko uses command `portrait:Riko_Prism Heart` only after her first awakening.\n"
                "The document deliberately does not use a fixed Image Command heading."
            )
        },
    }


class _JsonRequest:
    def __init__(self, body):
        self._body = body

    async def json(self):
        return deepcopy(self._body)


class _InlineLlmQueue:
    def __init__(self):
        self.added = []
        self.progress_events = []
        self.items = []

    async def _notify_progress(self, item, detail):
        item.progress = float(detail.get("percentage") or 0)
        item.progress_detail = deepcopy(detail)
        self.progress_events.append({
            "item_id": item.id,
            "progress": item.progress,
            "detail": deepcopy(detail),
        })

    async def cancel_item(self, item_id):
        item = next((value for value in self.items if value.id == item_id), None)
        if item is None or item.status != "pending":
            return False
        item.status = "cancelled"
        return True

    async def add_item(
        self,
        item_type,
        label,
        params,
        priority=10,
        runtime_handler=None,
        **_kwargs,
    ):
        item = SimpleNamespace(
            id="visual-guide-queue",
            type=item_type,
            label=label,
            status="processing",
            params=deepcopy(params),
            progress=0.0,
            progress_detail={},
            completion_future=asyncio.get_running_loop().create_future(),
        )
        self.items.append(item)
        self.added.append({
            "item_type": item_type,
            "label": label,
            "params": deepcopy(params),
            "priority": priority,
        })
        try:
            result = await runtime_handler(item)
        except Exception as exc:
            item.completion_future.set_exception(exc)
        else:
            item.status = (
                "cancelled"
                if getattr(item, "_runtime_cancelled", False)
                else "completed"
            )
            item.completion_future.set_result(result)
        return item


@pytest.mark.asyncio
async def test_suggest_metadata_reads_the_whole_selected_prompt_and_does_not_save(monkeypatch):
    bot_mode = importlib.import_module("modes.bot_mode")
    llm_service = importlib.import_module("modes.llm_service")
    lighbd_service = importlib.import_module("modes.lighbd_service")
    cards = [_card("card_1", "카드 1", "Riko_magical_overcome_smile.webp")]
    data = _bot_data(cards)
    captured = {}
    detail_records = []
    queue = _InlineLlmQueue()

    async def fake_call(task_key, messages, **kwargs):
        captured["task_key"] = task_key
        captured["messages"] = messages
        raw = json.dumps(
            {
                "suggestions": [
                    {
                        "target_key": "0",
                        "aliases": ["Riko_Prism Heart", "Prism Heart"],
                        "selection_guide": (
                            "리코가 최초 각성을 마치고 Prism Heart 형태를 유지하는 동안 선택한다. "
                            "각성 이전이나 다른 형태일 때는 선택하지 않는다."
                        ),
                        "evidence": "원문의 Riko_Prism Heart 명령과 대표 이미지의 overcome 형태가 대응한다.",
                        "confidence": "high",
                    }
                ]
            },
            ensure_ascii=False,
        )
        valid, reason = kwargs["result_validator"](raw)
        assert valid, reason
        kwargs["metadata_sink"].update({
            "prompt_tokens": 120,
            "completion_tokens": 40,
            "tps": 20.0,
        })
        context = kwargs["execution_context"]
        kwargs["execution_observer"]({
            "type": "execution_complete",
            "llm_slot": "llm2",
            "phase": "primary",
            "execution_id": context.execution_id,
        })
        return raw

    monkeypatch.setattr(bot_mode, "_load_bot_data", lambda: data)
    monkeypatch.setattr(bot_mode, "_load_lb_extra", lambda _bot_name: [])
    monkeypatch.setattr(
        bot_mode,
        "_save_bot_data",
        lambda _value: pytest.fail("suggestion preview must not save bot.json"),
    )
    monkeypatch.setattr(llm_service, "callLLMTask", fake_call)
    monkeypatch.setattr(lighbd_service, "_log_lighbd_history", detail_records.append)

    manager = bot_mode.BotMode()
    manager.set_queue_manager(queue)

    response = await manager.handle_suggest_character_card_metadata(
        _JsonRequest(
            {
                "bot_name": "demo",
                "targets": [{"character": "Riko", "profile_id": "card_1"}],
            }
        )
    )
    payload = json.loads(response.text)

    assert response.status == 200
    assert payload["success"] is True
    assert payload["suggestions"][0]["aliases"] == [
        "Riko_Prism Heart",
        "Prism Heart",
    ]
    assert captured["task_key"] == "visual_profile_guide"
    assert queue.added[0]["item_type"] == "visual_profile_guide"
    assert queue.added[0]["params"]["target_count"] == 1
    assert len(detail_records) == 1
    assert detail_records[0]["task_key"] == "visual_profile_guide"
    assert detail_records[0]["status"] == "ok"
    assert detail_records[0]["llm_slot"] == "llm2"
    assert detail_records[0]["prompt_tokens"] == 120
    assert detail_records[0]["completion_tokens"] == 40
    assert detail_records[0]["queue_item_id"] == "visual-guide-queue"
    assert detail_records[0]["character"] == "Riko"
    assert detail_records[0]["profile_id"] == "card_1"
    assert detail_records[0]["profile_ids"] == ["card_1"]
    assert detail_records[0]["profile_count"] == 1
    assert detail_records[0]["character_index"] == 1
    assert detail_records[0]["character_count"] == 1
    prompt_text = "\n".join(str(message["content"]) for message in captured["messages"])
    assert "Arbitrary Picture Grammar" in prompt_text
    assert "fixed Image Command heading" in prompt_text
    assert "Riko_magical_overcome_smile.webp" in prompt_text
    assert "Default outfit evidence: school uniform" in prompt_text
    assert "Registered outfits:" not in prompt_text
    assert "fallback example, not a rule" in prompt_text
    assert "hardcoded keyword spotting" in prompt_text


@pytest.mark.asyncio
async def test_suggest_metadata_calls_llm_once_per_character(monkeypatch):
    bot_mode = importlib.import_module("modes.bot_mode")
    llm_service = importlib.import_module("modes.llm_service")
    lighbd_service = importlib.import_module("modes.lighbd_service")
    riko_cards = [
        _card("riko_1", "기본", "Riko_normal.webp"),
        _card("riko_2", "각성", "Riko_awakened.webp"),
    ]
    mina_cards = [_card("mina_1", "기본", "Mina_normal.webp")]
    data = _bot_data(riko_cards)
    data["bots"][0]["characters"].append({
        "name": "Mina",
        "visual_cards": deepcopy(mina_cards),
    })
    queue = _InlineLlmQueue()
    calls = []
    detail_records = []

    async def fake_call(task_key, messages, **kwargs):
        context = kwargs["execution_context"]
        metadata = context.metadata
        prompt = "\n".join(str(item["content"]) for item in messages)
        calls.append({
            "task_key": task_key,
            "metadata": deepcopy(metadata),
            "prompt": prompt,
        })
        raw = json.dumps({
            "suggestions": [
                {
                    "target_key": str(index),
                    "aliases": [f"alias-{profile_id}"],
                    "selection_guide": f"{profile_label} 프로필이 성립할 때 선택한다.",
                    "evidence": f"{profile_id} 프로필 근거",
                    "confidence": "high",
                }
                for index, (profile_id, profile_label) in enumerate(zip(
                    metadata["profile_ids"],
                    metadata["profile_labels"],
                ))
            ]
        }, ensure_ascii=False)
        valid, reason = kwargs["result_validator"](raw)
        assert valid, reason
        kwargs["execution_observer"]({
            "type": "execution_complete",
            "llm_slot": "llm1",
            "phase": "primary",
            "execution_id": context.execution_id,
        })
        return raw

    monkeypatch.setattr(bot_mode, "_load_bot_data", lambda: data)
    monkeypatch.setattr(bot_mode, "_load_lb_extra", lambda _bot_name: [])
    monkeypatch.setattr(llm_service, "callLLMTask", fake_call)
    monkeypatch.setattr(lighbd_service, "_log_lighbd_history", detail_records.append)

    manager = bot_mode.BotMode()
    manager.set_queue_manager(queue)
    response = await manager.handle_suggest_character_card_metadata(
        _JsonRequest({
            "bot_name": "demo",
            "targets": [
                {"character": "Riko", "profile_id": "riko_1"},
                {"character": "Mina", "profile_id": "mina_1"},
                {"character": "Riko", "profile_id": "riko_2"},
            ],
        })
    )
    payload = json.loads(response.text)

    assert response.status == 200
    assert payload["success"] is True
    assert [
        (item["character"], item["profile_id"])
        for item in payload["suggestions"]
    ] == [
        ("Riko", "riko_1"),
        ("Riko", "riko_2"),
        ("Mina", "mina_1"),
    ]
    assert len(calls) == 2
    assert queue.added[0]["params"]["target_count"] == 3
    assert queue.added[0]["params"]["character_call_count"] == 2
    assert calls[0]["task_key"] == "visual_profile_guide"
    assert calls[0]["metadata"]["character"] == "Riko"
    assert calls[0]["metadata"]["profile_ids"] == ["riko_1", "riko_2"]
    assert calls[0]["metadata"]["profile_count"] == 2
    assert calls[0]["metadata"]["character_index"] == 1
    assert calls[0]["metadata"]["character_count"] == 2
    assert "Riko_normal.webp" in calls[0]["prompt"]
    assert "Riko_awakened.webp" in calls[0]["prompt"]
    assert "Mina_normal.webp" not in calls[0]["prompt"]
    assert calls[1]["metadata"]["character"] == "Mina"
    assert calls[1]["metadata"]["profile_ids"] == ["mina_1"]
    assert calls[1]["metadata"]["profile_count"] == 1
    assert calls[1]["metadata"]["character_index"] == 2
    assert calls[1]["metadata"]["character_count"] == 2
    assert "Mina_normal.webp" in calls[1]["prompt"]
    assert "Riko_normal.webp" not in calls[1]["prompt"]
    assert "Riko_awakened.webp" not in calls[1]["prompt"]
    assert len(detail_records) == 2
    assert [record["character"] for record in detail_records] == ["Riko", "Mina"]
    assert detail_records[0]["profile_ids"] == ["riko_1", "riko_2"]
    assert detail_records[0]["target_count"] == 2
    assert detail_records[1]["profile_ids"] == ["mina_1"]
    assert detail_records[1]["target_count"] == 1
    assert [event["progress"] for event in queue.progress_events] == [
        0.0,
        50.0,
        50.0,
        100.0,
    ]
    assert [event["detail"]["stage"] for event in queue.progress_events] == [
        "processing",
        "completed",
        "processing",
        "completed",
    ]
    assert queue.progress_events[0]["detail"]["character"] == "Riko"
    assert queue.progress_events[0]["detail"]["current"] == 1
    assert queue.progress_events[0]["detail"]["total"] == 2
    assert len(queue.progress_events[1]["detail"]["suggestions"]) == 2
    assert queue.progress_events[2]["detail"]["character"] == "Mina"
    assert len(queue.progress_events[3]["detail"]["suggestions"]) == 1


@pytest.mark.asyncio
async def test_visual_guide_cancel_stops_active_stream_and_skips_remaining_characters(
    monkeypatch,
):
    bot_mode = importlib.import_module("modes.bot_mode")
    llm_service = importlib.import_module("modes.llm_service")
    lighbd_service = importlib.import_module("modes.lighbd_service")
    data = _bot_data([_card("riko_1", "기본", "Riko_normal.webp")])
    data["bots"][0]["characters"].append({
        "name": "Mina",
        "visual_cards": [_card("mina_1", "기본", "Mina_normal.webp")],
    })
    queue = _InlineLlmQueue()
    detail_records = []
    calls = []
    cancel_calls = []
    stream_started = asyncio.Event()
    stream_released = asyncio.Event()

    async def fake_call(_task_key, _messages, **kwargs):
        calls.append(kwargs["execution_context"].metadata["character"])
        await kwargs["stream_observer"]({
            "type": "request_mode",
            "streaming": True,
            "llm_slot": "llm1",
        })
        await kwargs["stream_observer"]({
            "type": "stream_open",
            "stream_id": "visual-stream-1",
            "llm_slot": "llm1",
        })
        stream_started.set()
        await stream_released.wait()
        await kwargs["stream_observer"]({
            "type": "cancelled",
            "stream_id": "visual-stream-1",
            "llm_slot": "llm1",
        })
        context = kwargs["execution_context"]
        kwargs["execution_observer"]({
            "type": "execution_complete",
            "llm_slot": "llm1",
            "phase": "primary",
            "execution_id": context.execution_id,
        })
        return llm_service.ManualCancelledText("[LLM 실패] 사용자 중지")

    def fake_stream_control(stream_id, action):
        cancel_calls.append((stream_id, action))
        stream_released.set()
        return True, action

    monkeypatch.setattr(bot_mode, "_load_bot_data", lambda: data)
    monkeypatch.setattr(bot_mode, "_load_lb_extra", lambda _bot_name: [])
    monkeypatch.setattr(llm_service, "callLLMTask", fake_call)
    monkeypatch.setattr(llm_service, "request_stream_control", fake_stream_control)
    monkeypatch.setattr(lighbd_service, "_log_lighbd_history", detail_records.append)

    manager = bot_mode.BotMode()
    manager.set_queue_manager(queue)
    suggest_task = asyncio.create_task(
        manager.handle_suggest_character_card_metadata(
            _JsonRequest({
                "bot_name": "demo",
                "targets": [
                    {"character": "Riko", "profile_id": "riko_1"},
                    {"character": "Mina", "profile_id": "mina_1"},
                ],
            })
        )
    )
    try:
        await asyncio.wait_for(stream_started.wait(), timeout=2)
        assert len(queue.items) == 1
        cancel_response = await manager.handle_cancel_character_card_metadata_suggestion(
            _JsonRequest({"item_id": queue.items[0].id})
        )
        cancel_payload = json.loads(cancel_response.text)
        response = await asyncio.wait_for(suggest_task, timeout=2)
    finally:
        stream_released.set()

    payload = json.loads(response.text)
    assert cancel_response.status == 200
    assert cancel_payload["success"] is True
    assert cancel_payload["mode"] == "stream_cancel_requested"
    assert cancel_payload["stream_cancelled"] == 1
    assert cancel_calls == [("visual-stream-1", "cancel")]
    assert calls == ["Riko"]
    assert response.status == 200
    assert payload["success"] is True
    assert payload["cancelled"] is True
    assert payload["completed_character_count"] == 0
    assert payload["character_call_count"] == 2
    assert payload["suggestions"] == []
    assert queue.items[0].status == "cancelled"
    assert [event["detail"]["stage"] for event in queue.progress_events] == [
        "processing",
        "cancelling",
        "cancelled",
    ]
    assert len(detail_records) == 1
    assert detail_records[0]["status"] == "cancelled"


@pytest.mark.asyncio
async def test_visual_guide_cancel_waits_for_non_streaming_call_and_keeps_its_result(
    monkeypatch,
):
    bot_mode = importlib.import_module("modes.bot_mode")
    llm_service = importlib.import_module("modes.llm_service")
    lighbd_service = importlib.import_module("modes.lighbd_service")
    data = _bot_data([_card("riko_1", "기본", "Riko_normal.webp")])
    data["bots"][0]["characters"].append({
        "name": "Mina",
        "visual_cards": [_card("mina_1", "기본", "Mina_normal.webp")],
    })
    queue = _InlineLlmQueue()
    detail_records = []
    calls = []
    call_started = asyncio.Event()
    call_released = asyncio.Event()

    async def fake_call(_task_key, _messages, **kwargs):
        metadata = kwargs["execution_context"].metadata
        calls.append(metadata["character"])
        await kwargs["stream_observer"]({
            "type": "request_mode",
            "streaming": False,
            "llm_slot": "llm1",
        })
        call_started.set()
        await call_released.wait()
        raw = json.dumps({
            "suggestions": [{
                "target_key": "0",
                "aliases": ["Riko_Normal"],
                "selection_guide": "리코의 기본 프로필이 성립할 때 선택한다.",
                "evidence": "기본 형태 근거",
                "confidence": "high",
            }]
        }, ensure_ascii=False)
        valid, reason = kwargs["result_validator"](raw)
        assert valid, reason
        context = kwargs["execution_context"]
        kwargs["execution_observer"]({
            "type": "execution_complete",
            "llm_slot": "llm1",
            "phase": "primary",
            "execution_id": context.execution_id,
        })
        return raw

    monkeypatch.setattr(bot_mode, "_load_bot_data", lambda: data)
    monkeypatch.setattr(bot_mode, "_load_lb_extra", lambda _bot_name: [])
    monkeypatch.setattr(llm_service, "callLLMTask", fake_call)
    monkeypatch.setattr(lighbd_service, "_log_lighbd_history", detail_records.append)

    manager = bot_mode.BotMode()
    manager.set_queue_manager(queue)
    suggest_task = asyncio.create_task(
        manager.handle_suggest_character_card_metadata(
            _JsonRequest({
                "bot_name": "demo",
                "targets": [
                    {"character": "Riko", "profile_id": "riko_1"},
                    {"character": "Mina", "profile_id": "mina_1"},
                ],
            })
        )
    )
    try:
        await asyncio.wait_for(call_started.wait(), timeout=2)
        cancel_response = await manager.handle_cancel_character_card_metadata_suggestion(
            _JsonRequest({"item_id": queue.items[0].id})
        )
        cancel_payload = json.loads(cancel_response.text)
        call_released.set()
        response = await asyncio.wait_for(suggest_task, timeout=2)
    finally:
        call_released.set()

    payload = json.loads(response.text)
    assert cancel_response.status == 200
    assert cancel_payload["mode"] == "after_current"
    assert cancel_payload["stream_cancelled"] == 0
    assert calls == ["Riko"]
    assert response.status == 200
    assert payload["cancelled"] is True
    assert payload["completed_character_count"] == 1
    assert payload["character_call_count"] == 2
    assert [item["profile_id"] for item in payload["suggestions"]] == ["riko_1"]
    assert queue.items[0].status == "cancelled"
    assert [event["detail"]["stage"] for event in queue.progress_events] == [
        "processing",
        "cancelling",
        "completed",
        "cancelled",
    ]
    assert len(detail_records) == 1
    assert detail_records[0]["status"] == "ok"


@pytest.mark.asyncio
async def test_suggest_metadata_accepts_unsaved_modal_source_text(monkeypatch):
    bot_mode = importlib.import_module("modes.bot_mode")
    llm_service = importlib.import_module("modes.llm_service")
    lighbd_service = importlib.import_module("modes.lighbd_service")
    cards = [_card("card_1", "카드 1", "Riko_modal_source.webp")]
    data = _bot_data(cards)
    captured = {}
    queue = _InlineLlmQueue()

    async def fake_call(_task_key, messages, **kwargs):
        captured["prompt"] = "\n".join(str(item["content"]) for item in messages)
        raw = json.dumps(
            {
                "suggestions": [
                    {
                        "target_key": "0",
                        "aliases": ["modal-command"],
                        "selection_guide": "팝업에 붙여 넣은 지침이 성립할 때 선택한다.",
                        "evidence": "팝업 임시 원문",
                        "confidence": "medium",
                    }
                ]
            },
            ensure_ascii=False,
        )
        valid, reason = kwargs["result_validator"](raw)
        assert valid, reason
        return raw

    monkeypatch.setattr(bot_mode, "_load_bot_data", lambda: data)
    monkeypatch.setattr(bot_mode, "_load_lb_extra", lambda _bot_name: [])
    monkeypatch.setattr(llm_service, "callLLMTask", fake_call)
    monkeypatch.setattr(lighbd_service, "_log_lighbd_history", lambda _record: None)

    manager = bot_mode.BotMode()
    manager.set_queue_manager(queue)

    response = await manager.handle_suggest_character_card_metadata(
        _JsonRequest(
            {
                "bot_name": "demo",
                "source_text": "Completely different modal-only image grammar: modal-command",
                "targets": [{"character": "Riko", "profile_id": "card_1"}],
            }
        )
    )
    payload = json.loads(response.text)

    assert response.status == 200
    assert payload["source"] == {"preset": "팝업 임시 원문", "scope": "modal"}
    assert "Completely different modal-only image grammar" in captured["prompt"]
    assert "Arbitrary Picture Grammar" not in captured["prompt"]


@pytest.mark.asyncio
async def test_visual_guide_queue_failure_records_attempt_and_final_detail(monkeypatch):
    bot_mode = importlib.import_module("modes.bot_mode")
    llm_service = importlib.import_module("modes.llm_service")
    lighbd_service = importlib.import_module("modes.lighbd_service")
    cards = [_card("card_1", "카드 1", "Riko_invalid.webp")]
    data = _bot_data(cards)
    queue = _InlineLlmQueue()
    detail_records = []
    invalid_raw = '{"suggestions":[]}'

    async def fake_call(_task_key, _messages, **kwargs):
        valid, reason = kwargs["result_validator"](invalid_raw)
        assert valid is False
        kwargs["metadata_sink"].update({
            "prompt_tokens": 70,
            "completion_tokens": 5,
        })
        context = kwargs["execution_context"]
        kwargs["on_attempt_failure"]({
            "result": invalid_raw,
            "reason": reason,
            "phase": "primary",
            "slot": "llm3",
            "attempt": 1,
            "total_attempts": 1,
            "attempt_id": "visual-attempt-1",
            "elapsed": 0.25,
        })
        kwargs["execution_observer"]({
            "type": "execution_complete",
            "llm_slot": "llm3",
            "phase": "primary",
            "execution_id": context.execution_id,
        })
        return "[LLM 실패] visual_profile_guide primary 재시도 소진"

    monkeypatch.setattr(bot_mode, "_load_bot_data", lambda: data)
    monkeypatch.setattr(bot_mode, "_load_lb_extra", lambda _bot_name: [])
    monkeypatch.setattr(llm_service, "callLLMTask", fake_call)
    monkeypatch.setattr(lighbd_service, "_log_lighbd_history", detail_records.append)

    manager = bot_mode.BotMode()
    manager.set_queue_manager(queue)
    response = await manager.handle_suggest_character_card_metadata(
        _JsonRequest({
            "bot_name": "demo",
            "targets": [{"character": "Riko", "profile_id": "card_1"}],
        })
    )
    payload = json.loads(response.text)

    assert response.status == 422
    assert "캐릭터 호출 실패" in payload["error"]
    assert len(detail_records) == 2
    assert detail_records[0]["history_id"] == "visual-attempt-1"
    assert detail_records[0]["attempt"] == 1
    assert detail_records[0]["status"] == "error"
    assert detail_records[0]["output"] == invalid_raw
    assert detail_records[1]["status"] == "error"
    assert detail_records[1]["output"] == invalid_raw
    assert detail_records[1]["llm_slot"] == "llm3"
    assert "target_key" in detail_records[1]["error"]


@pytest.mark.asyncio
async def test_queue_manager_dispatches_visual_guide_runtime_handler():
    queue_manager = importlib.import_module("queue_manager")
    manager = queue_manager.QueueManager()
    item = SimpleNamespace(
        id="visual-runtime-item",
        type="visual_profile_guide",
        params={"bot_name": "demo"},
    )

    async def runtime_handler(received):
        assert received is item
        return {"success": True}

    item._runtime_handler = runtime_handler
    result = await manager._execute_item(item)

    assert result == {"success": True}


@pytest.mark.asyncio
async def test_queue_manager_preserves_partial_result_for_runtime_cancellation():
    queue_manager = importlib.import_module("queue_manager")
    manager = queue_manager.QueueManager()
    item = queue_manager.QueueItem(
        id="visual-runtime-cancelled",
        type="visual_profile_guide",
        label="프로필 선택 기준 중단 테스트",
    )
    partial_result = {
        "suggestions": [{"character": "Riko", "profile_id": "riko_1"}],
        "cancelled": True,
    }

    async def runtime_handler(_item):
        _item._runtime_cancelled = True
        _item._runtime_cancel_reason = "사용자 중단"
        _item._return_result_on_cancel = True
        return partial_result

    async def no_prune(_item):
        return None

    item._runtime_handler = runtime_handler
    item.completion_future = asyncio.get_running_loop().create_future()
    manager.items.append(item)
    manager._deferred_prune = no_prune

    await manager._run_item_pipeline(item, is_gpu=False)

    assert item.status == "cancelled"
    assert item.error == "사용자 중단"
    assert item.progress == 0.0
    assert await item.completion_future == partial_result


def test_visual_profile_guide_is_registered_in_routing_queue_and_frontend():
    root = importlib.import_module("pathlib").Path(__file__).resolve().parents[1]
    server_source = (root / "server.py").read_text(encoding="utf-8")
    frontend = (root / "frontend" / "index.html").read_text(encoding="utf-8")
    queue_manager = importlib.import_module("queue_manager")

    assert '"visual_profile_guide": _llm_route_defaults(json_mode=True)' in server_source
    assert "{ key: 'visual_profile_guide'" in frontend
    assert "visual_profile_guide" in queue_manager.LLM_TYPES
    assert 'visual_profile_guide: \'프로필 선택 기준\'' in frontend
    assert '"/api/bot_mode/character_cards/suggest_metadata/cancel"' in server_source


@pytest.mark.asyncio
async def test_apply_metadata_preserves_existing_fields_and_saves_once(monkeypatch):
    bot_mode = importlib.import_module("modes.bot_mode")
    cards = [
        _card(
            "card_1",
            "카드 1",
            "Riko_normal.webp",
            aliases=["manual alias"],
            guide="사람이 직접 작성한 선택 기준",
        ),
        _card("card_2", "카드 2", "Riko_awakened.webp"),
    ]
    data = _bot_data(cards)
    saved = []
    monkeypatch.setattr(bot_mode, "_load_bot_data", lambda: data)
    monkeypatch.setattr(bot_mode, "_load_lb_extra", lambda _bot_name: [])
    monkeypatch.setattr(bot_mode, "_save_bot_data", lambda value: saved.append(deepcopy(value)))

    response = await bot_mode.BotMode().handle_apply_character_card_metadata(
        _JsonRequest(
            {
                "bot_name": "demo",
                "overwrite": False,
                "items": [
                    {
                        "character": "Riko",
                        "profile_id": "card_1",
                        "aliases": ["generated normal"],
                        "selection_guide": "생성된 기본 기준",
                    },
                    {
                        "character": "Riko",
                        "profile_id": "card_2",
                        "aliases": ["Riko_Prism Heart"],
                        "selection_guide": "각성 형태가 유지되는 동안 선택한다.",
                    },
                ],
            }
        )
    )
    payload = json.loads(response.text)
    stored_cards = saved[0]["bots"][0]["characters"][0]["visual_cards"]

    assert response.status == 200
    assert payload["applied"] == 1
    assert payload["skipped"] == 1
    assert len(saved) == 1
    assert stored_cards[0]["aliases"] == ["manual alias"]
    assert stored_cards[0]["selection_guide"] == "사람이 직접 작성한 선택 기준"
    assert stored_cards[1]["aliases"] == ["Riko_Prism Heart"]
    assert stored_cards[1]["selection_guide"] == "각성 형태가 유지되는 동안 선택한다."


@pytest.mark.asyncio
async def test_apply_metadata_can_replace_reviewed_existing_values(monkeypatch):
    bot_mode = importlib.import_module("modes.bot_mode")
    cards = [
        _card(
            "card_1",
            "카드 1",
            "Riko_normal.webp",
            aliases=["old"],
            guide="old guide",
        )
    ]
    data = _bot_data(cards)
    saved = []
    monkeypatch.setattr(bot_mode, "_load_bot_data", lambda: data)
    monkeypatch.setattr(bot_mode, "_load_lb_extra", lambda _bot_name: [])
    monkeypatch.setattr(bot_mode, "_save_bot_data", lambda value: saved.append(deepcopy(value)))

    response = await bot_mode.BotMode().handle_apply_character_card_metadata(
        _JsonRequest(
            {
                "bot_name": "demo",
                "overwrite": True,
                "items": [
                    {
                        "character": "Riko",
                        "profile_id": "card_1",
                        "aliases": ["Riko_Normal"],
                        "selection_guide": "리코의 일반 형태일 때 선택한다.",
                    }
                ],
            }
        )
    )
    payload = json.loads(response.text)
    stored = saved[0]["bots"][0]["characters"][0]["visual_cards"][0]

    assert response.status == 200
    assert payload["applied"] == 1
    assert stored["aliases"] == ["Riko_Normal"]
    assert stored["selection_guide"] == "리코의 일반 형태일 때 선택한다."


def test_frontend_has_thumbnail_review_modal_and_explicit_apply_modes():
    frontend = (
        importlib.import_module("pathlib").Path(__file__).resolve().parents[1]
        / "frontend"
        / "index.html"
    ).read_text(encoding="utf-8")

    assert "openVisualGuideGeneratorModal" in frontend
    assert "overlay.id = 'visual-guide-generator-overlay'" in frontend
    assert "visual-guide-thumb" in frontend
    assert "LLM 제안" in frontend
    assert "판단 근거" in frontend
    assert "비어 있는 값만 채우기" in frontend
    assert "기존 값도 교체" in frontend
    assert "source_text: state.sourceText" in frontend
    assert "이 임시 원문은 시스템 프롬프트에 저장되지 않습니다" in frontend
    assert "_handleVisualGuideQueueProgress(data)" in frontend
    assert "_visualGuideMergeSuggestions(liveSuggestions)" in frontend
    assert "detail.phase !== 'visual_profile_guide'" in frontend
    assert "LLM 처리 중 ${progressPosition}" in frontend
    assert "d.phase === 'visual_profile_guide'" in frontend
    assert 'id="visual-guide-stop"' in frontend
    assert "stopVisualGuideGeneration()" in frontend
    assert "/api/bot_mode/character_cards/suggest_metadata/cancel" in frontend
    assert "/api/bot_mode/character_cards/suggest_metadata" in frontend
    assert "/api/bot_mode/character_cards/apply_metadata" in frontend
    assert "overlay.onclick" not in frontend[
        frontend.index("async function openVisualGuideGeneratorModal"):
        frontend.index("function closeVisualGuideGeneratorModal")
    ]


def test_visual_guide_generator_has_one_bot_sidebar_entry_point():
    frontend = (
        importlib.import_module("pathlib").Path(__file__).resolve().parents[1]
        / "frontend"
        / "index.html"
    ).read_text(encoding="utf-8")

    sidebar_start = frontend.index('<div id="bot-char-section"')
    sidebar_end = frontend.index('<div id="bot-one-click-workflow-group"', sidebar_start)
    sidebar = frontend[sidebar_start:sidebar_end]
    character_cards_start = frontend.index("async function renderBotCharacters()")
    character_cards_end = frontend.index("function renderBotCharLoraList", character_cards_start)
    character_cards = frontend[character_cards_start:character_cards_end]

    assert frontend.count("✨ 이미지 지침으로 자동 작성") == 1
    assert 'id="btn-visual-guide-generator"' in sidebar
    assert 'onclick="openVisualGuideGeneratorModal()"' in sidebar
    assert "openVisualGuideGeneratorModal(" not in character_cards
