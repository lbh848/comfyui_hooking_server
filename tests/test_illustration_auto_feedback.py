import pathlib
import sys
from types import SimpleNamespace

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import queue_manager as queue_manager_module
import server


def test_auto_feedback_llm_routing_is_registered_in_backend_and_frontend() -> None:
    route = server.DEFAULT_CONFIG["llm_routing"][
        server.ILLUSTRATION_AUTO_FEEDBACK_REVIEW_TASK_KEY
    ]
    assert route["json_mode"] is True

    frontend = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    assert "illustration_auto_feedback_review" in frontend
    assert "삽화 오토피드백 검수" in frontend
    assert "startIllustrationAutoFeedback()" in frontend
    assert 'id="illustration-auto-feedback-modal"' in frontend
    assert 'id="llm-edit-auto-feedback-current-direction"' in frontend
    assert "openIllustrationAutoFeedbackModal()" in frontend
    assert "closePromptModal();" in frontend
    assert "안전 중지" in frontend
    assert "모든 회차 이미지는 백업에 보존" in frontend


def test_auto_feedback_llm_phases_use_the_unified_llm_queue() -> None:
    assert "illustration_auto_feedback_llm" in queue_manager_module.LLM_TYPES
    assert (
        queue_manager_module.RESERVED_ILLUSTRATION_TYPE_ORDER[
            "illustration_auto_feedback_llm"
        ]
        == 2
    )


def test_auto_feedback_review_parser_requires_machine_consumed_fields() -> None:
    parsed, error = server._parse_auto_feedback_review(
        """
        {
          "achieved": false,
          "score": 72,
          "summary": "분위기는 맞지만 두 인물의 시선이 어긋납니다.",
          "remaining_gaps": "두 인물이 서로 바라봐야 합니다.",
          "next_direction": "두 인물의 얼굴과 시선을 서로 향하게 수정하세요."
        }
        """
    )
    assert error == ""
    assert parsed == {
        "achieved": False,
        "score": 72,
        "summary": "분위기는 맞지만 두 인물의 시선이 어긋납니다.",
        "remaining_gaps": "두 인물이 서로 바라봐야 합니다.",
        "next_direction": "두 인물의 얼굴과 시선을 서로 향하게 수정하세요.",
    }

    invalid, invalid_error = server._parse_auto_feedback_review(
        '{"achieved":"false","score":72,"summary":"x","next_direction":"y"}'
    )
    assert invalid is None
    assert "true/false" in invalid_error


@pytest.mark.asyncio
async def test_auto_feedback_review_records_retry_and_final_result_in_lb_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records: list[dict] = []
    widget_events: list[tuple[str, dict]] = []
    captured_call: dict = {}

    async def notify_frontend(event_type, data):
        widget_events.append((event_type, data))

    async def call_vision_result(_task_key, _messages, **kwargs):
        captured_call.update(kwargs)
        await kwargs["on_attempt_failure"]({
            "attempt_id": "attempt-1",
            "phase": "primary",
            "slot": "llm1",
            "attempt": 1,
            "total_attempts": 2,
            "elapsed": 0.2,
            "reason": "첫 응답 검증 실패",
            "raw_response": '{"achieved":"no"}',
        })
        raw = (
            '{"achieved":true,"score":96,"summary":"목표 달성",'
            '"remaining_gaps":"","next_direction":""}'
        )
        return SimpleNamespace(
            accepted=True,
            raw_response=raw,
            text=raw,
            final_slot="llm2",
            final_phase="fallback",
            reason="",
            exception=None,
        )

    monkeypatch.setattr(server, "notify_frontend", notify_frontend)
    monkeypatch.setattr(
        server.llm_service,
        "callLLMVisionTaskResult",
        call_vision_result,
    )
    monkeypatch.setattr(
        server,
        "_auto_feedback_slot_identity",
        lambda slot: (f"service-{slot}", f"model-{slot}"),
    )
    monkeypatch.setattr(
        server.lighbd_service,
        "_log_lighbd_history",
        lambda record: records.append(record),
    )

    job = {
        "job_id": "job-review",
        "goal": "밤 배경에서 두 인물이 서로 바라보며 웃게 해줘",
        "max_rounds": 3,
    }
    result = await server._execute_auto_feedback_review(
        SimpleNamespace(id="queue-review"),
        job,
        1,
        b"original",
        b"generated",
    )

    assert result["achieved"] is True
    assert len(captured_call["images"]) == 2
    assert [record["status"] for record in records] == ["error", "ok"]
    assert records[0]["execution_id"] == "attempt-1"
    assert records[0]["llm_slot"] == "llm1"
    assert records[1]["llm_slot"] == "llm2"
    assert records[1]["phase"] == "fallback"
    assert records[1]["service"] == "service-llm2"
    assert records[1]["model"] == "model-llm2"
    assert records[1]["task_key"] == "illustration_auto_feedback_review"
    assert records[1]["queue_item_id"] == "queue-review"
    assert records[1]["input"]
    assert records[1]["output"]
    assert [data["type"] for event, data in widget_events if event == "lighbd_llm_stream"] == [
        "start",
        "done",
    ]


@pytest.mark.asyncio
async def test_auto_feedback_preserves_failed_round_backups_until_goal_is_met(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    (tmp_path / "source.webp").write_bytes(b"original-image")
    monkeypatch.setattr(server, "WORKFLOW_BACKUP_DIR", str(tmp_path))

    async def notify_frontend(_event_type, _data):
        return None

    async def run_edit(_job, round_number, _edit_body):
        return {
            "plan": f"{round_number}회 구체적인 프롬프트 수정 계획",
            "positive": f"positive-{round_number}",
            "negative": f"negative-{round_number}",
        }

    generated_backups: list[str] = []

    async def run_regeneration(_job, round_number, _regenerate_body):
        backup_name = f"generated-{round_number}"
        generated_backups.append(backup_name)
        return {
            "image_bytes": f"image-{round_number}".encode(),
            "backup_name": backup_name,
            "provider": "comfy",
            "fallback_used": False,
        }

    async def run_review(_job, round_number, _original, _generated):
        if round_number == 1:
            return {
                "achieved": False,
                "score": 61,
                "summary": "첫 결과는 목표 미달입니다.",
                "remaining_gaps": "표정이 다릅니다.",
                "next_direction": "표정을 목표에 맞게 수정하세요.",
            }
        return {
            "achieved": True,
            "score": 94,
            "summary": "목표를 충족했습니다.",
            "remaining_gaps": "",
            "next_direction": "",
        }

    monkeypatch.setattr(server, "notify_frontend", notify_frontend)
    monkeypatch.setattr(server, "_run_auto_feedback_edit", run_edit)
    monkeypatch.setattr(server, "_run_auto_feedback_regeneration", run_regeneration)
    monkeypatch.setattr(server, "_run_auto_feedback_review", run_review)

    body = {
        "name": "source",
        "positive": "source positive",
        "negative": "source negative",
        "direction": "밤 배경에서 두 인물이 서로 바라보며 웃게 해줘",
        "max_rounds": 3,
    }
    job = server._create_auto_feedback_job(body)
    try:
        await server._run_illustration_auto_feedback_job(job["job_id"], body)

        assert job["status"] == "completed"
        assert job["achieved"] is True
        assert generated_backups == ["generated-1", "generated-2"]
        assert [entry["backup_name"] for entry in job["rounds"]] == [
            "generated-1",
            "generated-2",
        ]
        assert job["rounds"][0]["review"]["achieved"] is False
        assert job["rounds"][0]["edit_direction"] == "1회 구체적인 프롬프트 수정 계획"
        assert job["rounds"][1]["edit_direction"] == "2회 구체적인 프롬프트 수정 계획"
        assert "표정을 목표에 맞게 수정하세요." in job["rounds"][1]["llm_instruction"]
        assert job["current_direction"] == job["rounds"][1]["edit_direction"]
        assert job["best_backup_name"] == "generated-2"
        assert server._auto_feedback_public_job(job)["preserved_backup_names"] == [
            "generated-1",
            "generated-2",
        ]
    finally:
        server._illustration_auto_feedback_jobs.pop(job["job_id"], None)


@pytest.mark.asyncio
async def test_auto_feedback_max_rounds_selects_best_but_keeps_every_backup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    (tmp_path / "source.webp").write_bytes(b"original-image")
    monkeypatch.setattr(server, "WORKFLOW_BACKUP_DIR", str(tmp_path))

    async def notify_frontend(_event_type, _data):
        return None

    async def run_edit(_job, round_number, _edit_body):
        return {
            "plan": f"{round_number}회 구체적인 프롬프트 수정 계획",
            "positive": f"positive-{round_number}",
            "negative": "negative",
        }

    async def run_regeneration(_job, round_number, _regenerate_body):
        return {
            "image_bytes": f"image-{round_number}".encode(),
            "backup_name": f"kept-{round_number}",
        }

    scores = {1: 83, 2: 57}

    async def run_review(_job, round_number, _original, _generated):
        return {
            "achieved": False,
            "score": scores[round_number],
            "summary": f"{round_number}회 목표 미달",
            "remaining_gaps": "목표와 차이가 있습니다.",
            "next_direction": "남은 차이를 줄이세요.",
        }

    monkeypatch.setattr(server, "notify_frontend", notify_frontend)
    monkeypatch.setattr(server, "_run_auto_feedback_edit", run_edit)
    monkeypatch.setattr(server, "_run_auto_feedback_regeneration", run_regeneration)
    monkeypatch.setattr(server, "_run_auto_feedback_review", run_review)

    body = {
        "name": "source",
        "positive": "source positive",
        "negative": "source negative",
        "direction": "목표 장면",
        "max_rounds": 2,
    }
    job = server._create_auto_feedback_job(body)
    try:
        await server._run_illustration_auto_feedback_job(job["job_id"], body)

        assert job["status"] == "completed"
        assert job["achieved"] is False
        assert job["best_backup_name"] == "kept-1"
        assert job["best_score"] == 83
        assert server._auto_feedback_public_job(job)["preserved_backup_names"] == [
            "kept-1",
            "kept-2",
        ]
    finally:
        server._illustration_auto_feedback_jobs.pop(job["job_id"], None)


@pytest.mark.asyncio
async def test_auto_feedback_cancel_after_generation_keeps_generated_backup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    (tmp_path / "source.webp").write_bytes(b"original-image")
    monkeypatch.setattr(server, "WORKFLOW_BACKUP_DIR", str(tmp_path))

    async def notify_frontend(_event_type, _data):
        return None

    async def run_edit(_job, _round_number, _edit_body):
        return {
            "plan": "구체적인 프롬프트 수정 계획",
            "positive": "edited",
            "negative": "negative",
        }

    async def run_regeneration(job, _round_number, _regenerate_body):
        job["cancel_requested"] = True
        return {"image_bytes": b"kept-image", "backup_name": "kept-before-cancel"}

    async def unexpected_review(*_args, **_kwargs):
        raise AssertionError("중지 요청 뒤에는 비전 검수를 시작하면 안 됩니다")

    monkeypatch.setattr(server, "notify_frontend", notify_frontend)
    monkeypatch.setattr(server, "_run_auto_feedback_edit", run_edit)
    monkeypatch.setattr(server, "_run_auto_feedback_regeneration", run_regeneration)
    monkeypatch.setattr(server, "_run_auto_feedback_review", unexpected_review)

    body = {
        "name": "source",
        "positive": "source positive",
        "negative": "source negative",
        "direction": "목표 장면",
        "max_rounds": 3,
    }
    job = server._create_auto_feedback_job(body)
    try:
        await server._run_illustration_auto_feedback_job(job["job_id"], body)

        assert job["status"] == "cancelled"
        assert server._auto_feedback_public_job(job)["preserved_backup_names"] == [
            "kept-before-cancel"
        ]
    finally:
        server._illustration_auto_feedback_jobs.pop(job["job_id"], None)
