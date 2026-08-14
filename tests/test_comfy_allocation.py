from __future__ import annotations

import pytest

from comfy_allocation import (
    COMFY_TASK_KEYS,
    MODAL_COMFY_TARGET,
    VAST_COMFY_TARGET,
    ComfyTaskAllocationValidationError,
    normalize_comfy_task_allocations,
    normalize_comfy_task_modal_parallel,
    normalize_comfy_task_vast_parallel,
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


@pytest.mark.parametrize("value", (0, 4, True, 1.5, "02", "second"))
def test_allocation_rejects_invalid_instance_values(value) -> None:
    with pytest.raises(ComfyTaskAllocationValidationError):
        normalize_comfy_task_allocations({"illustration": value})


def test_every_local_task_can_use_third_instance_and_video_is_one_allocation() -> None:
    allocations = normalize_comfy_task_allocations(
        {"illustration": 3, "video_generation": 2}
    )

    assert allocations["illustration"] == 3
    assert allocations["video_generation"] == 2
    assert "video_t2v" not in allocations
    assert "video_i2v" not in allocations
    assert "video_first_last" not in allocations


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
        {1: False, 2: False, 3: False},
    ) == 2


def test_exactly_one_of_three_running_instances_is_used_as_local_fallback() -> None:
    allocations = normalize_comfy_task_allocations({"video_generation": 1})

    assert select_comfy_instance(
        allocations,
        "video_generation",
        {1: False, 2: False, 3: True},
    ) == 3


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
        "video_generation",
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
        "video_generation",
    ),
)
def test_vast_is_accepted_where_modal_is_supported(task_key: str) -> None:
    allocations = normalize_comfy_task_allocations({task_key: "VAST"})

    assert allocations[task_key] == VAST_COMFY_TARGET


@pytest.mark.parametrize(
    "task_key",
    ("tag_analysis", "outfit", "face_extract", "utility_debug"),
)
def test_vast_is_rejected_for_local_only_tasks(task_key: str) -> None:
    with pytest.raises(ComfyTaskAllocationValidationError, match="Vast"):
        normalize_comfy_task_allocations({task_key: "vast"})


def test_modal_parallel_is_allowed_with_local_or_vast_primary_target() -> None:
    local_allocations = normalize_comfy_task_allocations({"illustration": 2})
    modal_allocations = normalize_comfy_task_allocations({"illustration": "modal"})
    vast_allocations = normalize_comfy_task_allocations({"illustration": "vast"})

    local_parallel = normalize_comfy_task_modal_parallel(
        {"illustration": True, "tag_analysis": True},
        allocations=local_allocations,
    )
    modal_parallel = normalize_comfy_task_modal_parallel(
        {"illustration": True},
        allocations=modal_allocations,
    )
    vast_primary_parallel = normalize_comfy_task_modal_parallel(
        {"illustration": True},
        allocations=vast_allocations,
    )

    assert local_parallel["illustration"] is True
    assert local_parallel["tag_analysis"] is False
    assert modal_parallel["illustration"] is False
    assert vast_primary_parallel["illustration"] is True


def test_vast_parallel_is_allowed_with_local_or_modal_primary_target() -> None:
    local_allocations = normalize_comfy_task_allocations({"illustration": 2})
    modal_allocations = normalize_comfy_task_allocations({"illustration": "modal"})
    vast_allocations = normalize_comfy_task_allocations({"illustration": "vast"})

    local_parallel = normalize_comfy_task_vast_parallel(
        {"illustration": True, "tag_analysis": True},
        allocations=local_allocations,
    )
    modal_primary_parallel = normalize_comfy_task_vast_parallel(
        {"illustration": True},
        allocations=modal_allocations,
    )
    vast_parallel = normalize_comfy_task_vast_parallel(
        {"illustration": True},
        allocations=vast_allocations,
    )

    assert local_parallel["illustration"] is True
    assert local_parallel["tag_analysis"] is False
    assert modal_primary_parallel["illustration"] is True
    assert vast_parallel["illustration"] is False


@pytest.mark.parametrize(
    "normalizer",
    (normalize_comfy_task_modal_parallel, normalize_comfy_task_vast_parallel),
)
def test_remote_parallel_rejects_non_boolean_values(normalizer) -> None:
    allocations = normalize_comfy_task_allocations({"illustration": 1})

    with pytest.raises(ComfyTaskAllocationValidationError, match="bool"):
        normalizer(
            {"illustration": "true"},
            allocations=allocations,
        )


def test_local_instance_selection_rejects_modal_primary_target() -> None:
    allocations = normalize_comfy_task_allocations({"illustration": "modal"})

    with pytest.raises(ComfyTaskAllocationValidationError, match="Modal 전용"):
        select_comfy_instance(allocations, "illustration", {1: True, 2: True})


def test_vast_primary_allows_modal_parallel_and_rejects_local_selection() -> None:
    allocations = normalize_comfy_task_allocations({"illustration": "vast"})
    parallel = normalize_comfy_task_modal_parallel(
        {"illustration": True},
        allocations=allocations,
    )

    assert parallel["illustration"] is True
    with pytest.raises(ComfyTaskAllocationValidationError, match="Vast 전용"):
        select_comfy_instance(allocations, "illustration", {1: True, 2: True})
