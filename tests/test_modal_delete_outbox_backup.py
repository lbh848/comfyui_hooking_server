"""재시도 횟수만 올리는 저장은 백업을 남기지 않는다.

백업은 큐에 든 삭제 요청을 잃지 않으려는 장치다. attempts/last_error 를 올리는
저장은 아무것도 잃을 수 없는데, 그때도 백업을 남기면 재시도마다 파일이 하나씩
쌓인다 — 영구 실패 대상 하나로 1분에 12개가 생겼다.
"""

from __future__ import annotations

import json
from pathlib import Path

from modal_backend.service import ModalService


def _service(tmp_path: Path) -> ModalService:
    return ModalService(tmp_path, lambda: {"modal_enabled": True})


def _backups(tmp_path: Path) -> list[Path]:
    root = tmp_path / "backups" / "modal"
    return sorted(root.glob("*.json")) if root.is_dir() else []


def test_retry_bookkeeping_save_writes_no_backup(tmp_path: Path) -> None:
    service = _service(tmp_path)
    item = [{"remote_prefix": "SOYA_CHAR_LORA/x", "attempts": 0}]

    service._save_delete_outbox(item)
    service._save_delete_outbox(item)
    before = len(_backups(tmp_path))

    for attempts in range(1, 13):
        item[0]["attempts"] = attempts
        service._save_delete_outbox(item, backup=False)

    assert len(_backups(tmp_path)) == before, (
        "재시도 횟수만 올리는 저장은 백업을 남기면 안 된다"
    )
    saved = json.loads((tmp_path / "modal_lora_delete_outbox.json").read_text())
    assert saved[0]["attempts"] == 12, "그래도 내용은 저장돼야 한다"


def test_real_queue_change_still_backs_up(tmp_path: Path) -> None:
    service = _service(tmp_path)
    service._save_delete_outbox([{"remote_prefix": "a"}])
    before = len(_backups(tmp_path))

    service._save_delete_outbox([{"remote_prefix": "a"}, {"remote_prefix": "b"}])

    assert len(_backups(tmp_path)) == before + 1, (
        "큐가 실제로 바뀌는 저장은 백업을 남겨야 한다"
    )


def test_video_outbox_has_the_same_guard(tmp_path: Path) -> None:
    service = _service(tmp_path)
    item = [{"remote_artifacts": [{"remote_path": "v/1.mp4"}], "attempts": 0}]

    service._save_video_delete_outbox(item)
    service._save_video_delete_outbox(item)
    before = len(_backups(tmp_path))

    for attempts in range(1, 13):
        item[0]["attempts"] = attempts
        service._save_video_delete_outbox(item, backup=False)

    assert len(_backups(tmp_path)) == before
