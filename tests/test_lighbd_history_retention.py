from __future__ import annotations

import sys
from pathlib import Path


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
    assert (tmp_path / "요구사항" / "lighbd_history.jsonl.bak").is_file()
