import base64
import json
import pathlib
import sys
from types import SimpleNamespace

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modes import illustration_quality_inspection as quality


class QueueItem:
    id = "queue-inspection"


def _entries() -> list[dict]:
    return [
        {
            "slot": "2",
            "backup_name": "scene-2",
            "prompt_id": "prompt-2",
            "image_bytes": b"\x89PNG\r\n\x1a\nimage-two",
            "positive": "the character reaches toward a lantern; contact reads",
            "negative": "unmotivated costume change",
            "descriptor": {
                "kind": "scene",
                "scene_brief": (
                    "Ari reaches toward the lantern. The lower body is naturally "
                    "cropped by the frame and a foreground object occludes it."
                ),
                "camera": "medium side view",
                "characters": [{"name": "Ari", "outfit": "blue coat"}],
                "continuity_note": "The blue coat remains from the prior scene.",
            },
        },
        {
            "slot": 5,
            "backup_name": "scene-5",
            "prompt_id": "prompt-5",
            "image_bytes": b"RIFF1234WEBPimage-five",
            "positive": "the character opens the lantern by the river",
            "negative": "unmotivated costume change",
            "descriptor": {
                "kind": "scene",
                "scene": (
                    "At the river, Ari opens the lantern after removing the coat; "
                    "the change is supported by the narrative."
                ),
                "camera": "wide view",
                "anchor_before": "Ari arrived wearing the blue coat.",
                "anchor_after": "The lantern is open.",
            },
        },
    ]


def _config() -> dict:
    return {
        "llm_service": "provider-one",
        "llm_model": "model-one",
        "llm_service2": "provider-two",
        "llm_model2": "model-two",
        "llm_service3": "provider-three",
        "llm_model3": "model-three",
        "llm_service5": "provider-five",
        "llm_model5": "model-five",
    }


def _response(feedback_2: str = "No material issue observed.") -> str:
    return json.dumps(
        {
            "images": [
                {"slot": "5", "feedback": "The supported outfit change is coherent."},
                {"slot": "2", "feedback": feedback_2},
            ],
            "overall_feedback": "The two scenes preserve the narrative state.",
        }
    )


def _allowed_history_keys() -> set[str]:
    return {
        "task_key",
        "call_name",
        "history_id",
        "execution_id",
        "parent_execution_id",
        "llm_slot",
        "phase",
        "service",
        "model",
        "input",
        "output",
        "prompt_tokens",
        "completion_tokens",
        "elapsed",
        "tps",
        "status",
        "error",
        "queue_item_id",
        "inspection_session_id",
        "inspection_flow_run_id",
        "inspection_images",
    }


@pytest.mark.asyncio
async def test_two_actual_images_and_compact_history_are_preserved(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict = {}
    records: list[dict] = []
    flow_events: list[dict] = []

    async def fake_call(_task_key, messages, **kwargs):
        captured["messages"] = messages
        captured.update(kwargs)
        kwargs["metadata_sink"].update({"completion_tokens": 21, "prompt_tokens": 42})
        raw = _response("The required contact reads; a naturally cropped region is not missing.")
        return SimpleNamespace(
            accepted=True,
            raw_response=raw,
            text=raw,
            final_slot="llm2",
            final_phase="fallback",
            reason="",
            exception=None,
        )

    monkeypatch.setattr(quality.llm_service, "get_config", lambda: _config())
    monkeypatch.setattr(quality.llm_service, "callLLMVisionTaskResult", fake_call)
    monkeypatch.setattr(quality.lighbd_service, "_log_lighbd_history", records.append)
    monkeypatch.setattr(
        quality.illustration_flow,
        "llm_metadata",
        lambda **fields: flow_events.append(fields),
    )

    result = await quality.execute_inspection(
        QueueItem(),
        session_id="session-1",
        flow_run_id="flow-1",
        context=(
            "Ari crosses the river in a blue coat, reaches for the lantern, "
            "then removes the coat and opens it."
        ),
        entries=_entries(),
    )

    assert result["images"] == [
        {"slot": 2, "feedback": "The required contact reads; a naturally cropped region is not missing."},
        {"slot": 5, "feedback": "The supported outfit change is coherent."},
    ]
    assert captured["json_mode"] is True
    assert captured["result_validator"]
    assert len(captured["images"]) == 2
    assert base64.b64decode(captured["images"][0][0]) == _entries()[0]["image_bytes"]
    assert base64.b64decode(captured["images"][1][0]) == _entries()[1]["image_bytes"]
    assert [image[1] for image in captured["images"]] == ["image/png", "image/webp"]
    assert "SLOT 2" in captured["images"][0][2]
    assert "scene-5" in captured["images"][1][2]
    assert "Ari crosses the river" in captured["messages"][1]["content"]
    assert "naturally cropped" in captured["messages"][1]["content"]
    assert "removes the coat" in captured["messages"][1]["content"]
    assert flow_events == [{"input": captured["messages"], "status": "processing"}]

    assert len(records) == 1
    record = records[0]
    assert set(record) == _allowed_history_keys()
    assert record["status"] == "ok"
    assert record["parent_execution_id"] == "flow-1"
    assert record["llm_slot"] == "llm2"
    assert record["phase"] == "fallback"
    assert record["service"] == "provider-two"
    assert record["model"] == "model-two"
    assert record["queue_item_id"] == "queue-inspection"
    assert record["inspection_session_id"] == "session-1"
    assert record["inspection_flow_run_id"] == "flow-1"
    assert record["inspection_images"] == [
        {"slot": 2, "backup_name": "scene-2", "prompt_id": "prompt-2"},
        {"slot": 5, "backup_name": "scene-5", "prompt_id": "prompt-5"},
    ]
    history_input = json.dumps(record["input"], ensure_ascii=False)
    assert "image-two" not in history_input
    assert base64.b64encode(_entries()[0]["image_bytes"]).decode("ascii") not in history_input
    assert "raw_response" not in record
    assert "provider" not in record
    assert "state" not in record


def test_prompt_checks_semantic_opposites_without_age_policy():
    messages = quality._build_messages(
        (
            "The source supports a coat being removed after the river crossing. "
            "One image shows an exposed unclothed pelvic region that must be visible; "
            "another region is naturally cropped and occluded."
        ),
        _entries(),
    )
    system = messages[0]["content"]
    user = messages[1]["content"]
    assert "visible interaction and action match the narrative" in system
    assert "Do not evaluate hand or finger rendering or finger count" in system
    assert "required touch, grasp, or other interaction reads" in system
    assert "natural occlusion, crop, and framing" in system
    assert "fabric or another covering" in system
    assert "allowing clothing or condition changes" in system
    assert "Separate an observed image discrepancy from a speculative prompt cause" in system
    assert "smallest connected body fragment" in system
    assert "classify age" not in system.lower()
    assert "exposed unclothed pelvic region" in user
    assert "coat being removed" in user


def test_parser_requires_each_actual_slot_and_accepts_numeric_slot_strings():
    raw = {
        "images": [
            {"slot": "5", "feedback": "No material issue observed.", "score": 100},
            {"slot": "2", "feedback": "The scene is relevant."},
        ],
        "overall_feedback": "Continuity is coherent.",
        "unconsumed_diagnostic": "ignored",
    }
    assert quality._parse_inspection_response(raw, [2, 5]) == {
        "images": [
            {"slot": 2, "feedback": "The scene is relevant."},
            {"slot": 5, "feedback": "No material issue observed."},
        ],
        "overall_feedback": "Continuity is coherent.",
    }
    with pytest.raises(ValueError, match="slots"):
        quality._parse_inspection_response(
            {"images": [{"slot": "2", "feedback": "Only one."}], "overall_feedback": "x"},
            [2, 5],
        )
    with pytest.raises(ValueError, match="overall_feedback"):
        quality._parse_inspection_response(
            {"images": [{"slot": 2, "feedback": "x"}, {"slot": 5, "feedback": "y"}]},
            [2, 5],
        )


@pytest.mark.asyncio
async def test_retry_and_terminal_failure_audits_keep_active_slot_identity(
    monkeypatch: pytest.MonkeyPatch,
):
    records: list[dict] = []

    async def fake_call(_task_key, _messages, **kwargs):
        await kwargs["on_attempt_failure"](
            {
                "attempt_id": "inspection:primary:llm2:1",
                "phase": "primary",
                "slot": "llm2",
                "elapsed": 0.25,
                "reason": "invalid inspection JSON",
                "raw_response": "not-json",
            }
        )
        return SimpleNamespace(
            accepted=False,
            raw_response="not-json",
            text="not-json",
            final_slot="llm3",
            final_phase="fallback",
            reason="provider rejected the response",
            exception=None,
        )

    monkeypatch.setattr(quality.llm_service, "get_config", lambda: _config())
    monkeypatch.setattr(quality.llm_service, "callLLMVisionTaskResult", fake_call)
    monkeypatch.setattr(quality.lighbd_service, "_log_lighbd_history", records.append)

    with pytest.raises(RuntimeError, match="provider rejected"):
        await quality.execute_inspection(
            QueueItem(),
            session_id="session-retry",
            context="The source narrative stays in the river scene.",
            entries=_entries(),
        )

    assert [record["status"] for record in records] == ["error", "error"]
    retry, terminal = records
    assert set(retry) == _allowed_history_keys()
    assert retry["execution_id"] == "inspection:primary:llm2:1"
    assert retry["parent_execution_id"].startswith("illustration_quality_inspection:session-retry:")
    assert retry["llm_slot"] == "llm2"
    assert retry["phase"] == "primary"
    assert retry["service"] == "provider-two"
    assert retry["model"] == "model-two"
    assert "invalid inspection JSON" in retry["error"]
    assert terminal["execution_id"].startswith("illustration_quality_inspection:session-retry:")
    assert terminal["llm_slot"] == "llm3"
    assert terminal["phase"] == "fallback"
    assert terminal["service"] == "provider-three"
    assert terminal["model"] == "model-three"
    assert terminal["output"] == "not-json"
    assert terminal["inspection_images"]


@pytest.mark.asyncio
async def test_unexpected_call_failure_is_logged_without_image_payload(
    monkeypatch: pytest.MonkeyPatch,
):
    records: list[dict] = []

    async def fake_call(_task_key, _messages, **_kwargs):
        raise TimeoutError("vision provider timeout")

    monkeypatch.setattr(quality.llm_service, "get_config", lambda: _config())
    monkeypatch.setattr(quality.llm_service, "callLLMVisionTaskResult", fake_call)
    monkeypatch.setattr(quality.lighbd_service, "_log_lighbd_history", records.append)

    with pytest.raises(TimeoutError, match="vision provider timeout"):
        await quality.execute_inspection(
            QueueItem(),
            session_id="session-error",
            context="A complete narrative.",
            entries=_entries(),
        )

    assert len(records) == 1
    record = records[0]
    assert set(record) == _allowed_history_keys()
    assert record["phase"] == "primary"
    assert record["status"] == "error"
    assert record["llm_slot"] == "llm1"
    assert record["service"] == "provider-one"
    assert record["model"] == "model-one"
    assert "vision provider timeout" in record["error"]
    assert base64.b64encode(_entries()[0]["image_bytes"]).decode("ascii") not in json.dumps(
        record["input"], ensure_ascii=False
    )


def test_missing_image_and_slot_identity_safety(monkeypatch: pytest.MonkeyPatch):
    with pytest.raises(ValueError, match="no final image bytes"):
        quality._validate_entries([{"slot": "2", "image_bytes": b""}])
    with pytest.raises(ValueError, match="duplicate slot"):
        quality._validate_entries(
            [
                {"slot": "2", "image_bytes": b"a"},
                {"slot": 2, "image_bytes": b"b"},
            ]
        )

    monkeypatch.setattr(
        quality.llm_service,
        "get_config",
        lambda: {"llm_service": "only-one", "llm_model": "model-one"},
    )
    assert quality._slot_identity("llm1") == ("only-one", "model-one")
    assert quality._slot_identity("llm2") == ("", "")
