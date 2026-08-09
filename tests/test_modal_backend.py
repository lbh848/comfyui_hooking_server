from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from modal_backend.manifest import selected_install_plan, workflow_catalog
from modal_backend.service import cost_summary
from modal_backend.service import ModalService
from modal_backend.settings import ModalSettings
from modal_backend.workflow_assets import resolve_input_files, resolve_lora_files
from queue_manager import QueueItem, QueueManager


PROJECT_ROOT = Path(__file__).parents[1]


def test_modal_defaults_are_scale_to_zero_and_l4_budgeted() -> None:
    settings = ModalSettings.from_mapping({})
    cost = cost_summary(settings)

    assert settings.gpu == "L4"
    assert settings.max_concurrency == 2
    assert cost["assumptions"]["min_containers"] == 0
    assert cost["assumptions"]["scaledown_window_seconds"] == 15
    assert cost["l4_gpu_per_hour"] == pytest.approx(0.7992)
    assert cost["estimated_container_hours"] == pytest.approx(26.89)


@pytest.mark.parametrize(
    "config",
    [
        {"modal_gpu": "T4"},
        {"modal_max_concurrency": 0},
        {"modal_max_concurrency": 11},
        {"modal_profile": "bad profile"},
        {"modal_environment": "a" * 64},
    ],
)
def test_modal_settings_reject_invalid_values(config: dict) -> None:
    with pytest.raises(ValueError):
        ModalSettings.from_mapping(config)


def test_manifest_catalog_uses_workflow_dependencies() -> None:
    catalog = {item["id"]: item for item in workflow_catalog(PROJECT_ROOT)}

    assert catalog["comfy_workflow_source_path"]["size_gib"] == pytest.approx(8.13, abs=0.01)
    assert catalog["illustration_workflow_source_paths.v3"]["size_gib"] == pytest.approx(
        20.81, abs=0.01
    )
    assert catalog["qwen_edit_workflow_source_path"]["size_gib"] == pytest.approx(
        26.48, abs=0.01
    )


def test_selected_plan_requires_existing_bound_workflow(tmp_path: Path) -> None:
    workflow = tmp_path / "workflow.json"
    workflow.write_text("{}\n", encoding="utf-8")
    plan = selected_install_plan(
        PROJECT_ROOT,
        ["comfy_workflow_source_path"],
        {"comfy_workflow_source_path": str(workflow)},
    )

    assert plan["workflow_ids"] == ["comfy_workflow_source_path"]
    assert plan["model_count"] == 11
    assert plan["size_gib"] == pytest.approx(8.13, abs=0.01)
    assert plan["workflow_files"][0]["source_path"] == str(workflow.resolve())


def test_selected_plan_reports_missing_workflow_file() -> None:
    with pytest.raises(FileNotFoundError, match="로컬 워크플로우 파일이 없습니다"):
        selected_install_plan(
            PROJECT_ROOT,
            ["comfy_workflow_source_path"],
            {"comfy_workflow_source_path": ""},
        )


def test_workflow_assets_resolve_structured_lora_and_image_inputs(tmp_path: Path) -> None:
    lora_root = tmp_path / "models" / "loras" / "SOYA_CHAR_LORA"
    lora_file = lora_root / "Alice" / "Lora" / "hero.safetensors"
    lora_file.parent.mkdir(parents=True)
    lora_file.write_bytes(b"lora-data")
    input_root = tmp_path / "input"
    image_file = input_root / "refs" / "face.png"
    image_file.parent.mkdir(parents=True)
    image_file.write_bytes(b"png-data")
    workflow = {
        "1": {"class_type": "LoraLoader", "inputs": {"lora_name": "SOYA_CHAR_LORA/Alice/Lora/hero.safetensors"}},
        "2": {"class_type": "LoadImage", "inputs": {"image": "refs/face.png"}},
    }
    config = {"lora_load_path": str(lora_root), "comfy_input_dir": str(input_root)}

    loras = resolve_lora_files(workflow, config)
    inputs = resolve_input_files(workflow, config)

    assert loras[0]["source_path"] == str(lora_file)
    assert loras[0]["remote_path"] == "SOYA_CHAR_LORA/Alice/Lora/hero.safetensors"
    assert inputs == [{"source_path": str(image_file), "remote_name": "refs/face.png"}]


@pytest.mark.asyncio
async def test_disabled_modal_does_not_create_delete_outbox(tmp_path: Path) -> None:
    service = ModalService(tmp_path, lambda: {"modal_enabled": False})

    await service.enqueue_lora_delete("SOYA_CHAR_LORA/Alice/Lora/hero")

    assert not (tmp_path / "modal_lora_delete_outbox.json").exists()


def test_modal_enabled_routes_illustrations_away_from_local_gpu() -> None:
    manager = QueueManager()
    manager.get_config = lambda: {
        "modal_enabled": True,
        "modal_max_concurrency": 2,
        "illustration_provider": "comfy",
        "bot_selected": "test-bot",
    }
    illustration = QueueItem(id="a", type="illustration", label="a", params={})
    local_gpu = QueueItem(id="b", type="asset_generation", label="b", params={})

    assert manager._item_execution_area(illustration) == ("modal", "modal")
    assert manager._item_execution_area(local_gpu)[0] == "gpu"
    assert manager._target_modal_workers() == 2


@pytest.mark.asyncio
async def test_modal_workers_run_two_illustrations_in_parallel() -> None:
    manager = QueueManager()
    manager.get_config = lambda: {
        "modal_enabled": True,
        "modal_max_concurrency": 2,
        "illustration_provider": "comfy",
        "bot_selected": "test-bot",
    }
    both_started = asyncio.Event()
    active = 0
    peak = 0

    async def execute(_item):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        if active == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=1)
        active -= 1
        return {"ok": True}

    manager._execute_item = execute
    manager.items = [
        QueueItem(id="one", type="illustration", label="one", params={}),
        QueueItem(id="two", type="illustration", label="two", params={}),
    ]
    try:
        await manager._ensure_modal_workers()
        deadline = asyncio.get_running_loop().time() + 2
        while any(item.status in ("pending", "processing") for item in manager.items):
            if asyncio.get_running_loop().time() > deadline:
                raise TimeoutError("Modal 병렬 큐 테스트 제한 시간 초과")
            await asyncio.sleep(0.01)
        assert peak == 2
        assert all(item.status == "completed" for item in manager.items)
    finally:
        tasks = list(manager._modal_worker_tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
