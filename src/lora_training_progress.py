"""Measure LoRA training speed from step progress, excluding process startup."""
from __future__ import annotations

import math


class LoraTrainingProgress:
    def __init__(self):
        self._start: tuple[int, float] | None = None
        self._last: tuple[int, int, float] | None = None

    def update(self, progress: dict) -> dict:
        detail = dict(progress)
        if detail.get("phase") != "training":
            # Remote workers interleave standard ComfyUI progress events.
            if detail.get("phase") == "preparing":
                self._start = None
                self._last = None
            return detail

        step = detail.get("step")
        total = detail.get("total")
        elapsed = detail.get("elapsed_sec")
        detail.update(elapsed_sec=None, seconds_per_step=None, remaining_min=None)
        if (type(step) is not int or step < 0
                or type(total) is not int or total <= 0
                or type(elapsed) not in (int, float)
                or not math.isfinite(elapsed) or elapsed < 0):
            return detail

        # The trainer's clock includes model loading and cache preparation.
        # Anchor to its first step event, using its clock to avoid network delays.
        # Cache/load progress also uses counters; restart when the actual
        # training counter begins or its total changes.
        if (self._start is None or self._last is None
                or step < self._last[0] or total != self._last[1]
                or elapsed < self._last[2]):
            self._start = (step, elapsed)
        self._last = (step, total, elapsed)
        start_step, start_elapsed = self._start
        elapsed -= start_elapsed
        completed_steps = step - start_step
        detail["elapsed_sec"] = elapsed
        if completed_steps > 0 and elapsed > 0:
            seconds_per_step = elapsed / completed_steps
            detail["seconds_per_step"] = seconds_per_step
            detail["remaining_min"] = seconds_per_step * max(0, total - step) / 60
        return detail
