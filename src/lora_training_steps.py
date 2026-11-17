"""Shared step settings and image scheduling for LoRA training and previews."""
from __future__ import annotations
import math


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


def prepare_training_images(images: list, settings: dict, *, default_mode="per_image", default_step=50) -> tuple[list, int, int]:
    """Return source images, per-image steps and the exact training step limit.

    Repetition belongs to the trainer, not to exported files or image tensors.
    Total mode keeps every source image once, regardless of the step budget.
    """
    steps = training_step_settings(settings, default_mode=default_mode, default_step=default_step)
    from lora_training_workflow import training_workflow_epoch_settings
    epoch_settings = training_workflow_epoch_settings(settings)
    if epoch_settings.get("max_train_epochs", 0):
        total_steps = epoch_settings["max_train_epochs"] * math.ceil(len(images) / epoch_settings["gradient_accumulation_steps"])
        return images, 1, total_steps
    if steps["step_mode"] == "per_image":
        step = steps["step_per_image"]
        return images, step, len(images) * step
    return images, 1, steps["total_steps"] if images else 0
