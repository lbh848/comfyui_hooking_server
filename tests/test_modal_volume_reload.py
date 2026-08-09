from __future__ import annotations

from pathlib import Path

import pytest

from modal_backend import volume_reload


class FakeVolume:
    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.calls = 0

    def reload(self) -> None:
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError("there are open files preventing the operation")


def test_reload_volume_retries_transient_failures_with_backoff(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    now = [100.0]
    sleeps: list[float] = []
    volume = FakeVolume(failures=2)

    monkeypatch.setattr(volume_reload.time, "monotonic", lambda: now[0])

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    monkeypatch.setattr(volume_reload.time, "sleep", fake_sleep)

    volume_reload.reload_volume_with_retry(
        volume,
        label="loras",
        timeout_seconds=10.0,
        initial_delay_seconds=0.25,
        max_delay_seconds=1.0,
    )

    captured = capsys.readouterr()
    assert volume.calls == 3
    assert sleeps == [0.25, 0.5]
    assert captured.out.count("Volume reload 실패") == 2
    assert "Volume reload 재시도 성공: label=loras, attempt=3" in captured.out
    assert captured.err.count("Traceback (most recent call last)") == 2


def test_reload_volume_reraises_after_timeout(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    now = [200.0]
    sleeps: list[float] = []
    volume = FakeVolume(failures=100)

    monkeypatch.setattr(volume_reload.time, "monotonic", lambda: now[0])

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    monkeypatch.setattr(volume_reload.time, "sleep", fake_sleep)

    with pytest.raises(RuntimeError, match="open files"):
        volume_reload.reload_volume_with_retry(
            volume,
            label="models",
            timeout_seconds=0.5,
            initial_delay_seconds=0.25,
            max_delay_seconds=0.25,
        )

    captured = capsys.readouterr()
    assert volume.calls == 3
    assert sleeps == [0.25, 0.25]
    assert "Volume reload 재시도 제한시간 초과" in captured.out
    assert "label=models, attempts=3" in captured.out
    assert captured.err.count("Traceback (most recent call last)") == 3


def test_modal_worker_uses_bounded_reload_helper_for_all_request_paths() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "modal_backend" / "modal_app.py"
    ).read_text(encoding="utf-8")

    assert source.count(
        'reload_volume_with_retry(models_volume, label="models")'
    ) == 2
    assert source.count(
        'reload_volume_with_retry(loras_volume, label="loras")'
    ) == 2
    assert "models_volume.reload()" not in source
    assert "loras_volume.reload()" not in source
