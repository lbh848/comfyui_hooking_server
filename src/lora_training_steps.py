"""Shared step settings and image scheduling for LoRA training and previews."""
from __future__ import annotations

import random


def training_step_settings(settings: dict, *, default_mode="per_image", default_step=50) -> dict:
    step_mode = settings.get("step_mode", default_mode)
    if step_mode not in ("per_image", "total"):
        raise ValueError("STEP_PER_IMAGE 또는 전체 STEP을 선택하세요.")
    step_per_image = settings.get("step_per_image", default_step)
    # Existing style projects stored their total count under step_per_image.
    total_steps = settings.get("total_steps", step_per_image)
    for label, value in (("STEP_PER_IMAGE", step_per_image), ("전체 STEP", total_steps)):
        if type(value) not in (int, float) or not 1 <= value <= 9007199254740991 or value != int(value):
            raise ValueError(f"{label} 값을 1 이상의 정수로 입력하세요.")
    return {"step_mode": step_mode, "step_per_image": int(step_per_image), "total_steps": int(total_steps)}


def prepare_training_images(images: list, settings: dict, *, default_mode="per_image", default_step=50) -> tuple[list, int]:
    """Reuse style LoRA's full passes + sampled remainder for an exact total.

    A fixed sampling seed keeps previews, captions and exported images aligned.
    Per-image mode leaves the original dataset and repeat count unchanged.
    """
    steps = training_step_settings(settings, default_mode=default_mode, default_step=default_step)
    if steps["step_mode"] == "per_image":
        return images, steps["step_per_image"]
    if not images:
        return [], 1
    full, remainder = divmod(steps["total_steps"], len(images))
    return images * full + random.Random(0).sample(images, remainder), 1
