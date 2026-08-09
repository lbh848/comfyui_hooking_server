"""Modal Volume reload 재시도 유틸리티."""

from __future__ import annotations

import time
import traceback
from typing import Protocol


class ReloadableVolume(Protocol):
    def reload(self) -> None: ...


def reload_volume_with_retry(
    volume: ReloadableVolume,
    *,
    label: str,
    timeout_seconds: float = 30.0,
    initial_delay_seconds: float = 0.25,
    max_delay_seconds: float = 2.0,
) -> None:
    """열린 파일로 인한 일시 충돌을 포함해 Volume reload를 제한시간 내 재시도한다."""
    deadline = time.monotonic() + max(0.0, float(timeout_seconds))
    delay_seconds = max(0.0, float(initial_delay_seconds))
    max_delay_seconds = max(delay_seconds, float(max_delay_seconds))
    attempt = 0

    while True:
        attempt += 1
        try:
            volume.reload()
            if attempt > 1:
                print(
                    f"[MODAL_COMFY] Volume reload 재시도 성공: "
                    f"label={label}, attempt={attempt}",
                    flush=True,
                )
            return
        except Exception as exc:
            remaining_seconds = deadline - time.monotonic()
            print(
                f"[MODAL_COMFY] Volume reload 실패: label={label}, "
                f"attempt={attempt}, error={type(exc).__name__}: {exc}",
                flush=True,
            )
            traceback.print_exc()
            if remaining_seconds <= 0:
                print(
                    f"[MODAL_COMFY] Volume reload 재시도 제한시간 초과: "
                    f"label={label}, attempts={attempt}, "
                    f"timeout_seconds={timeout_seconds}",
                    flush=True,
                )
                raise

            sleep_seconds = min(delay_seconds, remaining_seconds)
            print(
                f"[MODAL_COMFY] Volume reload 재시도 대기: label={label}, "
                f"attempt={attempt}, sleep_seconds={sleep_seconds:.2f}",
                flush=True,
            )
            time.sleep(sleep_seconds)
            delay_seconds = min(delay_seconds * 2, max_delay_seconds)
