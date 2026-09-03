"""CoreML provider 인지.

macOS onnxruntime 은 CoreMLExecutionProvider 를 노출하는데 코드가 그것을 몰라
장치 목록이 auto/cpu 뿐이었다. 다만 auto 의 우선순위에는 넣지 않는다 — 미지원
연산 폴백과 FP16 수치 차이로 기존 결과를 조용히 바꿀 수 있어서다.
"""

from modes import onnx_execution


def _devices(monkeypatch, providers):
    monkeypatch.setattr(
        onnx_execution, "installed_providers", lambda: set(providers)
    )
    return {item["key"]: item for item in onnx_execution.list_devices()}


def test_coreml_is_listed_when_available(monkeypatch):
    devices = _devices(monkeypatch, {"CPUExecutionProvider", "CoreMLExecutionProvider"})
    assert "coreml0" in devices
    assert devices["coreml0"]["provider"] == "CoreMLExecutionProvider"


def test_coreml_is_absent_without_the_provider(monkeypatch):
    assert "coreml0" not in _devices(monkeypatch, {"CPUExecutionProvider"})


def test_cpu_follows_coreml_so_unsupported_ops_fall_back(monkeypatch):
    monkeypatch.setattr(
        onnx_execution,
        "installed_providers",
        lambda: {"CPUExecutionProvider", "CoreMLExecutionProvider"},
    )
    names = [
        p[0] if isinstance(p, tuple) else p
        for p in onnx_execution.providers_for("coreml0")
    ]
    assert names[0] == "CoreMLExecutionProvider"
    assert names[-1] == "CPUExecutionProvider"


def test_auto_does_not_pick_coreml(monkeypatch):
    """수치 차이로 기존 결과가 조용히 바뀔 수 있어 명시 선택일 때만 쓴다."""
    monkeypatch.setattr(
        onnx_execution,
        "installed_providers",
        lambda: {"CPUExecutionProvider", "CoreMLExecutionProvider"},
    )
    assert onnx_execution.auto_device_key() != "coreml0"


def test_coreml_session_counts_as_accelerated():
    class _Session:
        def get_providers(self):
            return ["CoreMLExecutionProvider", "CPUExecutionProvider"]

    assert onnx_execution.session_uses_gpu(_Session()) is True
