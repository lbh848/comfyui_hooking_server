"""기동 시 작업 배분 프리플라이트.

배분된 로컬 인스턴스가 응답하지 않으면 실제로 생성을 눌러 봐야만 알 수 있었다.
진단이 목적이라 실패해도 기동을 막지 않아야 한다.
"""

import pytest

import server


class _Probe:
    """포트 점검을 결정적으로 만든다 — 실제 소켓을 쓰면 이 머신에서 무엇이 떠
    있느냐에 따라 결과가 달라진다."""

    def __init__(self, connectable):
        self.connectable = connectable

    def __call__(self, *args, **kwargs):
        return self

    def settimeout(self, _seconds):
        return None

    def connect_ex(self, address):
        return 0 if address[1] in self.connectable else 1

    def close(self):
        return None


@pytest.fixture
def preflight(monkeypatch):
    def _run(allocations, *, open_ports=()):
        monkeypatch.setitem(
            server.app_config, "comfy_task_allocations", allocations
        )
        monkeypatch.setattr(server.socket, "socket", _Probe(set(open_ports)))
        return server._comfy_allocation_preflight()

    return _run


def test_unreachable_local_instance_is_reported(preflight):
    findings = preflight({"illustration": 1}, open_ports=())
    reported = {f["task"] for f in findings}
    assert "illustration" in reported
    entry = next(f for f in findings if f["task"] == "illustration")
    assert entry["remote_capable"] is True


def test_reachable_instance_is_not_reported(preflight):
    findings = preflight({"illustration": 1}, open_ports=(8188,))
    assert "illustration" not in {f["task"] for f in findings}


def test_remote_allocation_is_not_probed(preflight):
    """원격 배분은 로컬 포트가 없는 게 정상이다."""
    findings = preflight({"illustration": "modal"}, open_ports=())
    assert "illustration" not in {f["task"] for f in findings}


def test_broken_config_does_not_raise(preflight):
    assert preflight(object(), open_ports=()) == []


def test_logging_wrapper_swallows_failure(monkeypatch):
    """진단이 기동을 막으면 안 된다."""

    def _boom():
        raise RuntimeError("probe exploded")

    monkeypatch.setattr(server, "_comfy_allocation_preflight", _boom)
    server._log_comfy_allocation_preflight()
