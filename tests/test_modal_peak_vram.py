import subprocess
from types import SimpleNamespace

from modal_backend.modal_app import _sample_gpu_memory


class _ThreeSamples:
    def __init__(self) -> None:
        self.sample = 0

    def is_set(self) -> bool:
        return self.sample >= 3

    def wait(self, _seconds: float) -> None:
        self.sample += 1


def test_gpu_sampler_keeps_peak_and_tolerates_a_failed_probe(monkeypatch) -> None:
    outputs = iter(
        [
            SimpleNamespace(stdout="120, 24576\n"),
            RuntimeError("nvidia-smi unavailable once"),
            SimpleNamespace(stdout="380, 24576\n"),
        ]
    )

    def fake_run(*_args, **_kwargs):
        result = next(outputs)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(subprocess, "run", fake_run)
    stats: dict = {}

    _sample_gpu_memory(_ThreeSamples(), stats)

    assert stats == {"peak_mib": 380, "total_mib": 24576}
