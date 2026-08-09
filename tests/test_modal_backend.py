from __future__ import annotations

import asyncio
import json
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
    assert settings.scaledown_window_seconds == 15
    assert settings.status_refresh_seconds == 5
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
        {"modal_scaledown_window_seconds": 1},
        {"modal_scaledown_window_seconds": 1201},
        {"modal_status_refresh_seconds": 1},
        {"modal_status_refresh_seconds": 61},
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


@pytest.mark.asyncio
async def test_disabled_modal_status_skips_all_remote_account_and_billing_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = ModalService(tmp_path, lambda: {"modal_enabled": False})

    async def unexpected_account_check(_settings: ModalSettings) -> bool:
        raise AssertionError("Modal OFF 상태에서는 계정 CLI를 실행하면 안 됩니다.")

    async def unexpected_billing(*_args, **_kwargs) -> dict:
        raise AssertionError("Modal OFF 상태에서는 청구 API를 실행하면 안 됩니다.")

    monkeypatch.setattr(service, "account_connected", unexpected_account_check)
    monkeypatch.setattr(service, "_billing_for_settings", unexpected_billing)

    status = await service.status(include_runtime=True)

    assert status["connected"] is False
    assert status["connection_checked"] is False
    assert status["runtime"] == {"available": False, "reason": "disabled"}
    assert status["billing"]["available"] is False
    assert status["billing"]["reason"] == "disabled"
    assert status["billing"]["cache_seconds"] == 60
    with pytest.raises(RuntimeError, match="Modal 사용을 먼저 켜고 저장"):
        await service.billing(force_refresh=True)


@pytest.mark.asyncio
async def test_modal_billing_uses_sixty_second_cache_and_force_refresh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = ModalService(
        tmp_path,
        lambda: {
            "modal_enabled": True,
            "modal_monthly_credit_usd": 30,
        },
    )
    billing_calls = 0

    async def connected(_settings: ModalSettings) -> bool:
        return True

    async def run_command(args: list[str], **_kwargs) -> tuple[int, str, str]:
        nonlocal billing_calls
        assert args[-3:] == ["billing", "summary", "--json"]
        billing_calls += 1
        return (
            0,
            json.dumps(
                {
                    "metered_cost": "7.5",
                    "billed_cost": "4.0",
                    "adjustments": {
                        "plan_credits": "-3.25",
                        "free_volume_discount": "-0.25",
                    },
                    "metered_cost_breakdown": {
                        "gpu": "7.25",
                        "memory": "0.25",
                    },
                }
            ),
            "",
        )

    monkeypatch.setattr(service, "account_connected", connected)
    monkeypatch.setattr(service, "_run_command", run_command)

    first = await service.billing()
    second = await service.billing()
    forced = await service.billing(force_refresh=True)

    assert billing_calls == 2
    assert first["cached"] is False
    assert second["cached"] is True
    assert second["cache_seconds"] == 60
    assert forced["cached"] is False
    assert first["summary"]["metered_cost"] == "7.5"
    assert first["summary"]["adjustment_total"] == "-3.50"
    assert first["summary"]["billed_cost"] == "4.0"
    assert first["summary"]["configured_credit"] == "30.0"
    assert first["summary"]["remaining_credit_estimate"] == "22.5"


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


@pytest.mark.asyncio
async def test_managed_workflow_run_tracks_result_without_local_comfy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow = tmp_path / "workflow-api.json"
    workflow.write_text(
        json.dumps(
            {
                "nodes": [{"id": 1, "type": "EmptyLatentImage"}],
                "links": [],
            }
        ),
        encoding="utf-8",
    )
    config = {
        "modal_enabled": True,
        "modal_max_concurrency": 2,
        "comfy_workflow_source_path": str(workflow),
    }
    service = ModalService(PROJECT_ROOT, lambda: config)

    async def connected(_settings: ModalSettings) -> bool:
        return True

    converted = {
        "1": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": 64, "height": 64, "batch_size": 1},
        }
    }

    async def client_action(
        _settings: ModalSettings,
        action: str,
        *,
        timeout: float,
        **payload,
    ) -> dict:
        assert action == "convert_workflow"
        assert timeout == 960
        assert payload["workflow"]["nodes"][0]["type"] == "EmptyLatentImage"
        return converted

    async def generated(actual_workflow: dict, *, timeout_seconds: int = 3300):
        assert timeout_seconds == 3300
        assert actual_workflow == converted
        return b"png", {
            "prompt_id": "prompt-1",
            "content_type": "image/png",
            "lora_sync": {"uploaded": 0},
        }

    monkeypatch.setattr(service, "account_connected", connected)
    monkeypatch.setattr(service, "_run_client_action", client_action)
    monkeypatch.setattr(service, "generate", generated)

    state = await service.start_workflow_run("comfy_workflow_source_path")
    await service._workflow_run_tasks[state["job_id"]]

    completed = service.workflow_run_status(state["job_id"])
    image, content_type = service.workflow_run_image(state["job_id"])
    assert completed["state"] == "completed"
    assert completed["result_available"] is True
    assert "image_bytes" not in completed
    assert image == b"png"
    assert content_type == "image/png"
