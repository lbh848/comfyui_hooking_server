import asyncio
from io import BytesIO
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import server
from modes import llm_prompt_edit, multi_char_mask


def _layout():
    return {
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


def _snapshot():
    return multi_char_mask.normalize_multi_char_snapshot({
        "enable": True,
        "character_order": ["Left", "Right"],
        "layout": _layout(),
        "mask_location": "region_mask",
    })


def _positive(snapshot, *, scene="original scene", fingerprint=None):
    payload = {
        "enable": True,
        "char_num": 2,
        "char_name_list": ["Left", "Right"],
        "mask_fingerprint": fingerprint or snapshot["mask_fingerprint"],
    }
    return "\n".join([
        "[ANIMA_CONTENT]",
        scene,
        "[MULTI_CHAR]",
        json.dumps(payload, ensure_ascii=False),
        "[HRF_ACTIVATE]",
        "false",
    ])


def _regional_positive(snapshot):
    payload = {
        "enable": True,
        "char_num": 2,
        "char_name_list": ["Left", "Right"],
        "char_inform": ["Left, old pose", "Right, old pose"],
        "char_trigger_list": [["Left"], ["Right"]],
        "background_trigger_list": [],
        "background_prompt": "old rooftop",
        "composition_prompt": "two people standing apart",
        "mask_fingerprint": snapshot["mask_fingerprint"],
        "shared_tag": {
            "before_char": ["artist tag", "quality tag", "old rooftop"],
            "after_char": [],
        },
    }
    return "\n".join([
        "[ANIMA_QUALITY]",
        "quality tag",
        "[ANIMA_ARTIST]",
        "artist tag",
        "[ANIMA_CONTENT]",
        "Left, old pose | Right, old pose",
        "[ANIMA_ALL]",
        "Left, old pose | Right, old pose",
        "[SDXL_QUALITY]",
        "quality tag",
        "[SDXL_ARTIST]",
        "artist tag",
        "[SDXL]",
        "Left, old pose | Right, old pose",
        "[MULTI_CHAR]",
        json.dumps(payload, ensure_ascii=False),
        "[HRF_ACTIVATE]",
        "false",
    ])


def _write_backup_files(root: Path, name: str, positive: str, snapshot: dict):
    workflow = {
        "nodes": [
            {"title": "긍정프롬프트", "widgets_values": [positive]},
            {"title": "부정프롬프트", "widgets_values": ["negative"]},
        ]
    }
    (root / f"{name}.json").write_text(
        json.dumps(workflow, ensure_ascii=False),
        encoding="utf-8",
    )
    (root / f"{name}_info.json").write_text(
        json.dumps({
            "provider": "comfy",
            "illustration_multi_char": snapshot,
        }, ensure_ascii=False),
        encoding="utf-8",
    )


class _JsonRequest:
    def __init__(self, payload):
        self._payload = payload

    async def json(self):
        return self._payload


def _completed_queue_item(item_id="test-regeneration-item"):
    future = asyncio.get_running_loop().create_future()
    future.set_result({"success": True, "generation_time": 1.0})
    return SimpleNamespace(id=item_id, completion_future=future)


@pytest.mark.asyncio
async def test_save_backup_persists_normalized_multi_char_snapshot(tmp_path, monkeypatch):
    snapshot = _snapshot()
    positive = _positive(snapshot)
    raw_context = {
        "enable": True,
        "character_order": ["Left", "Right"],
        "layout": _layout(),
        "mask_location": "region_mask",
    }
    output = BytesIO()
    Image.new("RGB", (8, 8), "white").save(output, format="PNG")

    monkeypatch.setattr(server, "WORKFLOW_BACKUP_DIR", str(tmp_path))
    monkeypatch.setattr(server, "current_original_workflow", {
        "nodes": [
            {"title": "긍정프롬프트", "widgets_values": [""]},
            {"title": "부정프롬프트", "widgets_values": [""]},
        ]
    })
    monkeypatch.setattr(server, "current_api_workflow", {})
    monkeypatch.setattr(server, "current_conversion_info", {})
    monkeypatch.setattr(server, "cleanup_backups", lambda: None)
    monkeypatch.setattr(server, "_invalidate_backup_filter_cache", lambda: None)

    async def ignore_notify(*args, **kwargs):
        return None

    monkeypatch.setattr(server, "notify_frontend", ignore_notify)

    backup_name, _ = await server.save_backup(
        output.getvalue(),
        "multi-save-test",
        positive,
        "negative",
        illustration_multi_char=raw_context,
    )

    info = json.loads(
        (tmp_path / f"{backup_name}_info.json").read_text(encoding="utf-8")
    )
    assert info["illustration_multi_char"] == snapshot


@pytest.mark.asyncio
async def test_regenerate_api_passes_saved_multi_char_snapshot_to_gpu_queue(
    tmp_path,
    monkeypatch,
):
    name = "saved-multi"
    snapshot = _snapshot()
    positive = _positive(snapshot)
    _write_backup_files(tmp_path, name, positive, snapshot)
    captured = {}

    async def fake_add_item(item_type, label, params, priority=0):
        captured.update({
            "item_type": item_type,
            "params": params,
            "priority": priority,
        })
        return _completed_queue_item()

    monkeypatch.setattr(server, "WORKFLOW_BACKUP_DIR", str(tmp_path))
    monkeypatch.setattr(server.queue_manager, "add_item", fake_add_item)

    response = await server.handle_api_regenerate(_JsonRequest({"name": name}))

    response_payload = json.loads(response.text)
    assert response.status == 202
    assert response_payload["queued"] is True
    assert response_payload["job_id"] == "test-regeneration-item"
    assert captured["item_type"] == "regenerate"
    assert captured["params"]["illustration_multi_char"] == snapshot
    assert captured["params"]["regeneration_request_id"] == response_payload["request_id"]


@pytest.mark.asyncio
async def test_modified_regenerate_keeps_mask_and_rejects_structural_changes(
    tmp_path,
    monkeypatch,
):
    name = "modified-multi"
    snapshot = _snapshot()
    source_positive = _positive(snapshot)
    _write_backup_files(tmp_path, name, source_positive, snapshot)
    queued = []

    async def fake_add_item(item_type, label, params, priority=0):
        queued.append(params)
        return _completed_queue_item()

    monkeypatch.setattr(server, "WORKFLOW_BACKUP_DIR", str(tmp_path))
    monkeypatch.setattr(server.queue_manager, "add_item", fake_add_item)

    valid_response = await server.handle_api_reschedule_with_modified_prompt(
        _JsonRequest({
            "name": name,
            "positive": _positive(snapshot, scene="edited clothes and expression"),
            "negative": "negative",
        })
    )
    invalid_response = await server.handle_api_reschedule_with_modified_prompt(
        _JsonRequest({
            "name": name,
            "positive": _positive(
                snapshot,
                scene="edited scene",
                fingerprint="0" * 64,
            ),
            "negative": "negative",
        })
    )

    assert valid_response.status == 202
    assert queued[0]["illustration_multi_char"] == snapshot
    assert invalid_response.status == 409
    assert len(queued) == 1


@pytest.mark.asyncio
async def test_regenerate_api_returns_after_queue_registration_before_completion(
    tmp_path,
    monkeypatch,
):
    name = "pending-regeneration"
    snapshot = _snapshot()
    _write_backup_files(tmp_path, name, _positive(snapshot), snapshot)
    completion_future = asyncio.get_running_loop().create_future()
    notifications = []

    async def fake_add_item(_item_type, _label, _params, priority=0):
        assert priority == 0
        return SimpleNamespace(
            id="pending-regeneration-item",
            completion_future=completion_future,
        )

    async def fake_notify(event_type, data=None):
        notifications.append((event_type, data or {}))

    monkeypatch.setattr(server, "WORKFLOW_BACKUP_DIR", str(tmp_path))
    monkeypatch.setattr(server.queue_manager, "add_item", fake_add_item)
    monkeypatch.setattr(server, "notify_frontend", fake_notify)

    response = await asyncio.wait_for(
        server.handle_api_regenerate(_JsonRequest({"name": name})),
        timeout=0.5,
    )
    payload = json.loads(response.text)
    active_tasks = list(server._regeneration_background_tasks)

    assert response.status == 202
    assert payload["queued"] is True
    assert payload["job_id"] == "pending-regeneration-item"
    assert completion_future.done() is False
    assert active_tasks

    completion_future.set_result({"success": True, "generation_time": 0.1})
    await asyncio.gather(*active_tasks)

    assert notifications[-1][0] == "regeneration_completed"
    assert notifications[-1][1]["request_id"] == payload["request_id"]


def test_llm_edit_syncs_scene_into_multi_char_conditioning_without_changing_mask():
    snapshot = _snapshot()
    positive = _regional_positive(snapshot)
    blocks = llm_prompt_edit.parse_blocks(positive)
    parsed = {
        "plan": "배경과 표정을 수정",
        "scene_setup": "sunset beach, warm lighting",
        "scene_char": "Left, smiling, waving | Right, laughing, hands on hips",
        "scene_supplement": "two people facing each other at the same fixed positions",
    }

    messages = llm_prompt_edit.build_llm_messages(
        "해변에서 웃게 바꿔줘",
        "old scene",
        "old scene",
        multi_char_payload=json.loads(blocks["MULTI_CHAR"]),
    )
    reassembled, _ = llm_prompt_edit.reassemble(
        positive,
        blocks,
        {"anima": set(), "sdxl": set()},
        parsed,
    )
    payload = multi_char_mask.extract_multi_char_prompt_payload(reassembled)

    assert "exactly 2 non-empty character blocks" in messages[-1]["content"]
    assert payload["char_inform"] == [
        "Left, smiling, waving",
        "Right, laughing, hands on hips",
    ]
    assert payload["background_prompt"] == "sunset beach, warm lighting"
    assert payload["composition_prompt"].startswith("two people facing")
    assert payload["shared_tag"]["before_char"][-1] == "sunset beach, warm lighting"
    assert payload["mask_fingerprint"] == snapshot["mask_fingerprint"]

    valid, reason = llm_prompt_edit.validate_multi_char_edit_result(
        {"scene_char": "only one combined block"},
        payload,
    )
    assert valid is False
    assert "블록 수" in reason
