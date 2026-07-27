"""브라우저 LLM E2E용 부작용 격리 서버.

프로덕션 app/router/frontend는 그대로 사용하되 시작 시 데이터 백업·워크플로
갱신·공지 갱신·자동 브라우저 열기를 비활성화하고 LLM 로그 경로만 .work 아래로
돌린다. config.json과 운영 히스토리는 읽기만 한다.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from aiohttp import web

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import server
from modes import lighbd_service, llm_service


async def _noop_async(*_args, **_kwargs):
    return None


def _noop_sync(*_args, **_kwargs):
    return None


def main() -> None:
    work_dir = Path(
        os.environ.get(
            "LLM_E2E_WORK_DIR",
            str(PROJECT_ROOT / ".work" / "llm_e2e_live"),
        )
    )
    log_dir = work_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    llm_service.LOG_DIR = str(log_dir)
    llm_service.HISTORY_PATH = str(log_dir / "llm_history.jsonl")
    llm_service.HISTORY_BACKUP_DIR = str(work_dir / "backup")
    llm_service.HISTORY_BACKUP_PATH = str(
        work_dir / "backup" / "llm_history.jsonl.bak"
    )
    lighbd_service.LIGHBD_HISTORY_PATH = str(
        log_dir / "lighbd_history.jsonl"
    )

    server._backup_data_on_startup = _noop_sync
    server.update_workflow_if_needed = _noop_async
    server.refresh_noti_cache = _noop_async
    server.webbrowser.open = _noop_sync
    server.app_config["character_maker_rag_autostart"] = False
    server.embedding_service.update_config = _noop_sync
    server.embedding_service._load_profiles_from_file = _noop_sync

    server.init_queue_manager()
    port = int(os.environ.get("LLM_E2E_PORT", "8189"))
    print(f"[LLM_E2E_LIVE] starting port={port}", flush=True)
    web.run_app(
        server.app,
        host="127.0.0.1",
        port=port,
        print=None,
        handle_signals=False,
    )


if __name__ == "__main__":
    main()
