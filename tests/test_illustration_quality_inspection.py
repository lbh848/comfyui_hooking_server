import base64
import io
import json
import pathlib
import subprocess
import sys
from types import SimpleNamespace

import pytest
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modes import illustration_quality_inspection as quality


class QueueItem:
    id = "queue-inspection"


def test_illustration_flow_javascript_has_valid_syntax() -> None:
    result = subprocess.run(
        ["node", "--check", str(ROOT / "frontend" / "illustration_flow.js")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def _image_bytes(color: tuple[int, int, int], *, size=(2000, 1000)) -> bytes:
    output = io.BytesIO()
    image = Image.new("RGB", size, color)
    image.save(output, format="PNG")
    image.close()
    return output.getvalue()


def _entries() -> list[dict]:
    positive_two = "the character reaches toward a lantern; contact reads"
    negative = "unmotivated costume change"
    return [
        {
            "slot": "2",
            "backup_name": "scene-2",
            "prompt_id": "prompt-2",
            "image_bytes": _image_bytes((30, 60, 90)),
            "positive": positive_two,
            "negative": negative,
            "descriptor": {
                "kind": "scene",
                "scene_brief": (
                    "Ari reaches toward the lantern. The lower body is naturally "
                    "cropped by the frame and a foreground object occludes it."
                ),
                "camera": "medium side view",
                "characters": [{"name": "Ari", "outfit": "blue coat"}],
                "continuity_note": "The blue coat remains from the prior scene.",
                "raw_positive": positive_two,
                "raw_negative": negative,
            },
        },
        {
            "slot": 5,
            "backup_name": "scene-5",
            "prompt_id": "prompt-5",
            "image_bytes": _image_bytes((90, 60, 30), size=(1000, 1600)),
            "positive": "the character opens the lantern by the river",
            "negative": negative,
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
    }


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
        "inspection_scope",
        "inspection_session_id",
        "inspection_flow_run_id",
        "inspection_images",
    }


@pytest.mark.asyncio
async def test_image_review_delegates_original_bytes_to_shared_vision_transport(monkeypatch):
    captured: dict = {}
    records: list[dict] = []
    flow_events: list[dict] = []
    entry = _entries()[0]

    async def fake_call(_task_key, messages, **kwargs):
        captured["messages"] = messages
        captured.update(kwargs)
        kwargs["metadata_sink"].update({"completion_tokens": 21, "prompt_tokens": 42})
        raw = json.dumps({
            "feedback": "The required contact reads; the crop is natural.",
            "continuity_observation": "Ari wears a blue coat beside the lantern.",
        })
        assert kwargs["result_validator"](raw) == (True, "")
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
        scope=quality.IMAGE_SCOPE,
        session_id="session-1",
        flow_run_id="flow-1",
        context="Ari crosses the river in a blue coat and reaches for the lantern.",
        entries=[entry],
    )

    assert result == {
        "inspection_scope": "image",
        "slot": 2,
        "feedback": "The required contact reads; the crop is natural.",
        "continuity_observation": "Ari wears a blue coat beside the lantern.",
    }
    assert captured["json_mode"] is True
    assert "images" not in captured
    assert "image_b64" not in captured
    assert captured["image_mime"] == "application/octet-stream"
    assert captured["image_bytes"] == entry["image_bytes"]
    user_message = captured["messages"][1]["content"]
    assert "Ari crosses the river" in user_message
    assert "naturally cropped" in user_message
    assert user_message.count(entry["positive"]) == 1
    assert user_message.count(entry["negative"]) == 1
    assert len(flow_events) == 1
    assert flow_events[0]["input"] == captured["messages"]
    assert flow_events[0]["status"] == "processing"
    assert flow_events[0]["inspection_scope"] == "image"
    assert flow_events[0]["history_id"].startswith(
        "illustration_quality_inspection:image:session-1:2:"
    )
    assert flow_events[0]["inspection_images"] == [
        {"slot": 2, "backup_name": "scene-2", "prompt_id": "prompt-2"}
    ]

    assert len(records) == 1
    record = records[0]
    assert set(record) == _allowed_history_keys()
    assert record["status"] == "ok"
    assert record["inspection_scope"] == "image"
    assert record["parent_execution_id"] == "flow-1"
    assert record["llm_slot"] == "llm2"
    assert record["phase"] == "fallback"
    assert record["service"] == "provider-two"
    assert record["model"] == "model-two"
    assert record["inspection_images"] == [
        {"slot": 2, "backup_name": "scene-2", "prompt_id": "prompt-2"}
    ]
    history_text = json.dumps(record, ensure_ascii=False)
    assert base64.b64encode(entry["image_bytes"]).decode("ascii") not in history_text
    assert entry["image_bytes"].hex() not in history_text


@pytest.mark.asyncio
async def test_overall_review_delegates_one_png_contact_sheet_to_shared_transport(monkeypatch):
    captured: dict = {}
    records: list[dict] = []
    individual = [
        {
            "inspection_scope": "image",
            "slot": 2,
            "feedback": "No material issue observed.",
            "continuity_observation": "Ari wears the blue coat.",
        },
        {
            "inspection_scope": "image",
            "slot": 5,
            "feedback": "No material issue observed.",
            "continuity_observation": "Ari has removed the coat and opens the lantern.",
        },
    ]

    async def fake_call(_task_key, messages, **kwargs):
        captured["messages"] = messages
        captured.update(kwargs)
        raw = json.dumps({"overall_feedback": "The supported coat change is coherent."})
        return SimpleNamespace(
            accepted=True,
            raw_response=raw,
            text=raw,
            final_slot="llm1",
            final_phase="primary",
            reason="",
            exception=None,
        )

    monkeypatch.setattr(quality.llm_service, "get_config", lambda: _config())
    monkeypatch.setattr(quality.llm_service, "callLLMVisionTaskResult", fake_call)
    monkeypatch.setattr(quality.lighbd_service, "_log_lighbd_history", records.append)
    monkeypatch.setattr(quality.illustration_flow, "llm_metadata", lambda **_fields: None)

    result = await quality.execute_inspection(
        QueueItem(),
        scope=quality.OVERALL_SCOPE,
        session_id="session-overall",
        flow_run_id="flow-overall",
        context="Ari removes the blue coat before opening the lantern.",
        entries=_entries(),
        individual_results=individual,
    )

    assert result == {
        "inspection_scope": "overall",
        "overall_feedback": "The supported coat change is coherent.",
    }
    assert "images" not in captured
    assert "image_b64" not in captured
    assert captured["image_mime"] == "image/png"
    contact_sheet = captured["image_bytes"]
    with Image.open(io.BytesIO(contact_sheet)) as image:
        assert image.format == "PNG"
        assert image.size == (840, 500)
    user_message = captured["messages"][1]["content"]
    assert "Ari wears the blue coat." in user_message
    assert "Ari has removed the coat" in user_message
    assert "CONTACT SHEET LABEL SLOT 2" in user_message
    assert records[0]["inspection_scope"] == "overall"
    assert [ref["slot"] for ref in records[0]["inspection_images"]] == [2, 5]


def test_prompts_cover_requested_semantics_without_age_or_second_character_policy():
    messages = quality._build_image_messages(
        (
            "The source supports a coat being removed. One image shows an exposed "
            "unclothed pelvic region that must be visible; another is naturally cropped."
        ),
        _entries()[0],
    )
    system = messages[0]["content"]
    user = messages[1]["content"]
    assert "natural, coherent, immediately readable still" in system
    assert "source narrative is authoritative" in system
    assert "resolved wardrobe timeline or `outfit_state`" in system
    assert "not a literal pixel checklist" in system
    assert "what an unfamiliar viewer can read from the pixels" in system
    assert "primary action or interaction is unclear" in system
    assert "actor and receiver cannot be distinguished" in system
    assert "required local contact does not read" in system
    assert "established lying/seated/standing support" in system
    assert "changes without a narrative transition" in system
    assert "jolt, arch, tremor, or momentary stillness" in system
    assert "Respect natural crop and occlusion" in system
    assert "Do not evaluate finger count or fine hand rendering" in system
    assert "fused or detached limb or crop-edge shape" in system
    assert "complete subject reaction, pose, gaze, or aftermath" in system
    assert "only when that reaction or aftermath is itself the selected fact" in system
    assert "selected fact is an ongoing interaction" in system
    assert "required contact must be visibly present" in system
    assert "does not supply pixels" in system
    assert "face may remain the primary focus while one contact is secondary" in system
    assert "do not demand a contact-only insert" in system
    assert "Omission of an unnecessary anonymous fragment is not itself a defect" in system
    assert "required or present fragment fails when it is absent, vague, detached" in system
    assert "broad wall of skin" in system
    assert "disconnected body fragments entering from different edges" in system
    assert "minor literal pose or contact mismatch" in system
    assert "natural, semantically equivalent" in system
    assert "complete partner, identifiable face, silhouette" in system
    assert "remaining displaced garment that vanishes" in system
    assert "Do not infer a transition merely because two prompt descriptions differ" in system
    assert "do not penalize clothing outside the crop" in system
    assert "classify age" not in system.lower()
    assert "exposed unclothed pelvic region" in user

    overall = quality._build_overall_messages("source", _entries(), [])[0]["content"]
    assert "set-level outfit, identity, chronology, and story-state consistency" in overall
    assert "meaningful named-subject coverage" in overall
    assert "at least one distinct, independently readable current beat" in overall
    assert "duplicate or weaker moments of another subject" in overall
    assert "Do not demand equal counts" in overall
    assert "dialogue, an off-frame cause, or an unreadable candidate" in overall
    assert "do not invent a transition merely because adjacent images" in overall
    assert "unexplained disappearance, reappearance, or alternation" in overall
    assert "Do not repeat isolated anatomy or fine hand issues" in overall


def test_scope_specific_parsers_accept_fenced_json_and_reject_missing_fields():
    assert quality._parse_inspection_response(
        "```json\n{\"feedback\":\"Issue.\",\"continuity_observation\":\"Blue coat.\"}\n```",
        quality.IMAGE_SCOPE,
        2,
    ) == {
        "inspection_scope": "image",
        "slot": 2,
        "feedback": "Issue.",
        "continuity_observation": "Blue coat.",
    }
    assert quality._parse_inspection_response(
        {"overall_feedback": "Continuity is coherent.", "ignored": 1},
        quality.OVERALL_SCOPE,
    ) == {
        "inspection_scope": "overall",
        "overall_feedback": "Continuity is coherent.",
    }
    with pytest.raises(ValueError, match="continuity_observation"):
        quality._parse_inspection_response(
            {"feedback": "Only feedback."}, quality.IMAGE_SCOPE, 2
        )
    with pytest.raises(ValueError, match="overall_feedback"):
        quality._parse_inspection_response({"feedback": "x"}, quality.OVERALL_SCOPE)


@pytest.mark.asyncio
async def test_retry_and_terminal_failure_audits_keep_active_slot_identity(monkeypatch):
    records: list[dict] = []

    async def fake_call(_task_key, _messages, **kwargs):
        await kwargs["on_attempt_failure"]({
            "attempt_id": "inspection:primary:llm2:1",
            "phase": "primary",
            "slot": "llm2",
            "elapsed": 0.25,
            "reason": "invalid inspection JSON",
            "raw_response": "not-json",
        })
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
            scope=quality.IMAGE_SCOPE,
            session_id="session-retry",
            context="The source narrative stays in the river scene.",
            entries=[_entries()[0]],
        )

    assert [record["status"] for record in records] == ["error", "error"]
    retry, terminal = records
    assert set(retry) == _allowed_history_keys()
    assert retry["inspection_scope"] == "image"
    assert retry["execution_id"] == "inspection:primary:llm2:1"
    assert retry["parent_execution_id"].startswith(
        "illustration_quality_inspection:image:session-retry:2:"
    )
    assert retry["llm_slot"] == "llm2"
    assert retry["service"] == "provider-two"
    assert terminal["llm_slot"] == "llm3"
    assert terminal["service"] == "provider-three"
    assert terminal["output"] == "not-json"


@pytest.mark.asyncio
async def test_unexpected_call_failure_is_logged_without_image_payload(monkeypatch):
    records: list[dict] = []

    async def fake_call(_task_key, _messages, **_kwargs):
        raise TimeoutError("vision provider timeout")

    monkeypatch.setattr(quality.llm_service, "get_config", lambda: _config())
    monkeypatch.setattr(quality.llm_service, "callLLMVisionTaskResult", fake_call)
    monkeypatch.setattr(quality.lighbd_service, "_log_lighbd_history", records.append)

    with pytest.raises(TimeoutError, match="vision provider timeout"):
        await quality.execute_inspection(
            QueueItem(),
            scope=quality.IMAGE_SCOPE,
            session_id="session-error",
            context="A complete narrative.",
            entries=[_entries()[0]],
        )

    assert len(records) == 1
    record = records[0]
    assert set(record) == _allowed_history_keys()
    assert record["phase"] == "primary"
    assert record["status"] == "error"
    assert record["llm_slot"] == "llm1"
    assert "vision provider timeout" in record["error"]
    assert base64.b64encode(_entries()[0]["image_bytes"]).decode("ascii") not in json.dumps(
        record["input"], ensure_ascii=False
    )


def test_missing_image_duplicate_slot_and_scope_validation(monkeypatch):
    with pytest.raises(ValueError, match="no final image bytes"):
        quality._validate_entries(
            [{"slot": "2", "image_bytes": b""}], quality.IMAGE_SCOPE
        )
    with pytest.raises(ValueError, match="exactly one"):
        quality._validate_entries(_entries(), quality.IMAGE_SCOPE)
    with pytest.raises(ValueError, match="duplicate slot"):
        quality._validate_entries(
            [
                {"slot": "2", "image_bytes": b"a"},
                {"slot": 2, "image_bytes": b"b"},
            ],
            quality.OVERALL_SCOPE,
        )

    monkeypatch.setattr(
        quality.llm_service,
        "get_config",
        lambda: {"llm_service": "only-one", "llm_model": "model-one"},
    )
    assert quality._slot_identity("llm1") == ("only-one", "model-one")
    assert quality._slot_identity("llm2") == ("", "")
