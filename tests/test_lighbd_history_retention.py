from __future__ import annotations

import datetime as _dt
import asyncio
import sys
import types
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modes import lighbd_service


def test_multi_char_history_has_independent_retention_budget(tmp_path, monkeypatch):
    history_path = tmp_path / "logs" / "lighbd_history.jsonl"
    monkeypatch.setattr(lighbd_service, "LIGHBD_HISTORY_PATH", str(history_path))
    monkeypatch.setattr(lighbd_service, "BASE_DIR", str(tmp_path))
    monkeypatch.setattr(lighbd_service, "LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setattr(lighbd_service, "LIGHBD_GENERAL_HISTORY_MAX", 2)
    monkeypatch.setattr(lighbd_service, "LIGHBD_MULTI_CHAR_HISTORY_MAX", 3)
    monkeypatch.setattr(lighbd_service, "LIGHBD_HISTORY_MAX", 5)

    # _log_lighbd_history가 append 시점에 ts를 now()로 갱신하므로, 빡빡한 루프에서
    # 초 경계를 넘길 때 ts가 섞여 정렬 결과가 흔들리는 flake를 막기 위해
    # now()가 호출될 때마다 1초씩 증가하는 datetime을 주입한다. lighbd_service 안에서
    # 참조하는 datetime만 가짜로 바꾼다(전역 datetime 모듈은 그대로).
    class _StepDateTime:
        _counter = 0

        @classmethod
        def now(cls, tz=None):
            cls._counter += 1
            return _dt.datetime(2024, 1, 1, 0, 0, cls._counter)

    monkeypatch.setattr(
        lighbd_service,
        "datetime",
        types.SimpleNamespace(datetime=_StepDateTime),
    )

    records = [
        {"prompt_id": "general-1", "task_key": "illustration_call1"},
        {"prompt_id": "multi-1", "task_key": "illustration_multi_char_mask"},
        {"prompt_id": "general-2", "task_key": "illustration_call2"},
        {"prompt_id": "multi-2", "task_key": "illustration_multi_char_mask"},
        {"prompt_id": "general-3", "task_key": "illustration_call3"},
        {"prompt_id": "multi-3", "task_key": "illustration_multi_char_mask"},
        {"prompt_id": "general-4", "task_key": "illustration_call1"},
        {"prompt_id": "multi-4", "task_key": "illustration_multi_char_mask"},
    ]
    for record in records:
        lighbd_service._log_lighbd_history(record)

    saved = lighbd_service._load_lighbd_history(limit=5)
    assert [record["prompt_id"] for record in saved] == [
        "multi-2",
        "general-3",
        "multi-3",
        "general-4",
        "multi-4",
    ]
    assert (tmp_path / "logs" / "backups" / "lighbd_history.jsonl.bak").is_file()


@pytest.mark.asyncio
async def test_manual_parallel_race_records_winner_and_discarded_without_ok_duplicate(
    tmp_path,
    monkeypatch,
):
    history_path = tmp_path / "logs" / "lighbd_history.jsonl"
    monkeypatch.setattr(lighbd_service, "LIGHBD_HISTORY_PATH", str(history_path))
    monkeypatch.setattr(lighbd_service, "BASE_DIR", str(tmp_path))
    monkeypatch.setattr(lighbd_service, "LOG_DIR", str(tmp_path / "logs"))
    lighbd_service._MANUAL_RACE_SUCCESS_SUPPRESSIONS.clear()

    owner_task = asyncio.current_task()
    lighbd_service._log_manual_parallel_race({
        "race_id": "race-1",
        "owner_task_id": id(owner_task),
        "task_key": "unit_task",
        "call_name": "단위 요청",
        "service": "openai",
        "model": "model-1",
        "llm_slot": "llm1",
        "input": [{"role": "user", "content": "hello"}],
        "winner_stream_id": "replacement-id",
        "attempts": [
            {
                "stream_id": "original-id",
                "race_role": "original",
                "race_status": "lost",
                "outcome_kind": "cancelled",
                "text": "느린 부분",
                "completion_tokens": 2,
                "prompt_tokens": 4,
                "elapsed": 1.2,
                "tps": 1.7,
            },
            {
                "stream_id": "parallel-id",
                "race_role": "parallel",
                "outcome_kind": "cancelled",
                "text": "취소된 병렬 부분",
                "completion_tokens": 2,
                "prompt_tokens": 4,
                "elapsed": 0.5,
                "tps": 4.0,
                "error": "사용자 중지",
            },
            {
                "stream_id": "replacement-id",
                "race_role": "parallel",
                "outcome_kind": "success",
                "text": "빠른 완료",
                "completion_tokens": 3,
                "prompt_tokens": 4,
                "elapsed": 0.3,
                "tps": 10.0,
            },
        ],
    })
    lighbd_service._log_lighbd_history({
        "call_name": "단위 요청",
        "input": [{"role": "user", "content": "hello"}],
        "output": "빠른 완료",
        "status": "ok",
    })

    saved = lighbd_service._load_lighbd_history(limit=10)
    assert len(saved) == 3
    assert [record["status"] for record in saved] == [
        "race_lost",
        "cancelled",
        "race_won",
    ]
    assert saved[0]["race_role"] == "original"
    assert saved[1]["race_role"] == "parallel"
    assert saved[2]["race_role"] == "parallel"
    assert saved[2]["winner_stream_id"] == "replacement-id"
    assert len({record["history_id"] for record in saved}) == 3
    assert (tmp_path / "logs" / "backups" / "lighbd_history.jsonl.bak").is_file()
