from __future__ import annotations

import pytest

from comfy_allocation import (
    COMFY_TASK_KEYS,
    ComfyTaskAllocationValidationError,
    normalize_comfy_task_allocations,
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
