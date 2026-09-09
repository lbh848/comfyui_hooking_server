"""큐 밖에서 원격 실행 대상을 판단하는 경로.

CURRENT_COMFY_EXECUTION_TARGET 은 큐 워커 안에서만 설정된다. 모드의 워크플로우
준비·직접 제출은 큐 밖에서 일어나므로 그때는 비어 있고, 로컬 ComfyUI 가 없는
Modal 전용 구성에서는 로컬 포트 조회로 떨어져 실패했다 — 에셋 생성이 원천적으로
막혔다. UI 도 같은 엔드포인트를 쓰므로 macOS 한정 문제가 아니다.
"""

import pytest

import server
from comfy_allocation import MODAL_COMFY_TARGET


@pytest.fixture
def allocations(monkeypatch):
    def _set(value):
        monkeypatch.setitem(
            server.app_config, "comfy_task_allocations", {"illustration": value}
        )

    return _set


def test_queue_context_wins(allocations, monkeypatch):
    allocations(1)
    token = server.CURRENT_COMFY_EXECUTION_TARGET.set(MODAL_COMFY_TARGET)
    try:
        assert server.effective_execution_target("illustration") == MODAL_COMFY_TARGET
    finally:
        server.CURRENT_COMFY_EXECUTION_TARGET.reset(token)


def test_configured_remote_is_used_outside_the_queue(allocations):
    allocations(MODAL_COMFY_TARGET)
    assert server.effective_execution_target("illustration") == MODAL_COMFY_TARGET


def test_local_allocation_stays_local(allocations):
    """로컬 배분(인스턴스 번호)이면 빈 문자열 — 기존 경로가 그대로 돈다."""
    allocations(1)
    assert server.effective_execution_target("illustration") == ""


def test_broken_config_falls_back_to_local(monkeypatch):
    monkeypatch.setitem(server.app_config, "comfy_task_allocations", object())
    assert server.effective_execution_target("illustration") == ""
