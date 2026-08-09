from __future__ import annotations

import pytest

from comfy_allocation import (
    COMFY_TASK_KEYS,
    MODAL_COMFY_TARGET,
    ComfyTaskAllocationValidationError,
    normalize_comfy_task_allocations,
    normalize_comfy_task_modal_parallel,
    select_comfy_instance,
)


def test_allocation_defaults_every_task_to_first_instance() -> None:
    allocations = normalize_comfy_task_allocations({})

    assert tuple(allocations) == COMFY_TASK_KEYS
    assert set(allocations.values()) == {1}


def test_legacy_illustration_port_is_inherited_only_without_new_mapping() -> None:
    legacy = normalize_comfy_task_allocations(
        None,
        legacy_illustration_port=8187,
    )
    explicit = normalize_comfy_task_allocations(
        {"illustration": 1},
        legacy_illustration_port=8187,
    )

    assert legacy["illustration"] == 2
    assert legacy["restore_regenerate"] == 2
    assert explicit["illustration"] == 1
    assert explicit["restore_regenerate"] == 1


@pytest.mark.parametrize("value", (0, 3, True, 1.5, "02", "second"))
def test_allocation_rejects_invalid_instance_values(value) -> None:
    with pytest.raises(ComfyTaskAllocationValidationError):
        normalize_comfy_task_allocations({"illustration": value})


def test_exactly_one_running_instance_overrides_configured_allocation() -> None:
    allocations = normalize_comfy_task_allocations({"illustration": 2})

    assert select_comfy_instance(
        allocations,
        "illustration",
        {1: True, 2: False},
    ) == 1
    assert select_comfy_instance(
        allocations,
        "illustration",
        {1: False, 2: True},
    ) == 2


def test_configured_allocation_is_used_when_both_or_neither_are_running() -> None:
    allocations = normalize_comfy_task_allocations({"asset_generation": 2})

    assert select_comfy_instance(
        allocations,
        "asset_generation",
        {1: True, 2: True},
    ) == 2
    assert select_comfy_instance(
        allocations,
        "asset_generation",
        {1: False, 2: False},
    ) == 2


@pytest.mark.parametrize(
    "task_key",
    (
        "illustration",
        "restore_regenerate",
        "asset_generation",
        "qwen_edit",
        "asset_lora_training",
        "bot_lora_training",
        "instance_lora",
    ),
)
def test_modal_is_accepted_for_every_supported_task(task_key: str) -> None:
    allocations = normalize_comfy_task_allocations({task_key: "MODAL"})

    assert allocations[task_key] == MODAL_COMFY_TARGET


@pytest.mark.parametrize(
    "task_key",
    ("tag_analysis", "outfit", "face_extract", "utility_debug"),
)
def test_modal_is_rejected_for_local_only_tasks(task_key: str) -> None:
    with pytest.raises(ComfyTaskAllocationValidationError, match="Modal"):
        normalize_comfy_task_allocations({task_key: "modal"})


def test_modal_parallel_is_allowed_only_with_local_primary_target() -> None:
    local_allocations = normalize_comfy_task_allocations({"illustration": 2})
    modal_allocations = normalize_comfy_task_allocations({"illustration": "modal"})

    local_parallel = normalize_comfy_task_modal_parallel(
        {"illustration": True, "tag_analysis": True},
        allocations=local_allocations,
    )
    modal_parallel = normalize_comfy_task_modal_parallel(
        {"illustration": True},
        allocations=modal_allocations,
    )

    assert local_parallel["illustration"] is True
    assert local_parallel["tag_analysis"] is False
    assert modal_parallel["illustration"] is False


def test_local_instance_selection_rejects_modal_primary_target() -> None:
    allocations = normalize_comfy_task_allocations({"illustration": "modal"})

    with pytest.raises(ComfyTaskAllocationValidationError, match="Modal 전용"):
        select_comfy_instance(allocations, "illustration", {1: True, 2: True})
