from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from modal_backend import client_cli
from modal_backend.manifest import selected_install_plan, workflow_catalog
from modal_backend.service import cost_summary
from modal_backend.service import ModalService
from modal_backend.settings import ModalSettings
from modal_backend.workflow_assets import (
    build_local_model_index,
    resolve_explicit_input_files,
    resolve_input_files,
    resolve_lora_files,
    resolve_workflow_model_files,
)
from modes.asset_tool_mode import AssetToolMode
from queue_manager import QueueItem, QueueManager


PROJECT_ROOT = Path(__file__).parents[1]


def _modal_test_project(tmp_path: Path) -> tuple[Path, Path]:
    project_root = tmp_path / "project"
    manifest_target = (
        project_root / "comfy_installer" / "resources" / "install_manifest.json"
    )
    manifest_target.parent.mkdir(parents=True)
    manifest_target.write_text(
        (
            PROJECT_ROOT
            / "comfy_installer"
            / "resources"
            / "install_manifest.json"
        ).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    user_root = project_root / "comfy" / "user" / "default" / "workflows" / "SOYA_USER"
    user_root.mkdir(parents=True)
    (project_root / "comfy" / "models").mkdir(parents=True)
    return project_root, user_root


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


def test_manifest_catalog_does_not_use_local_install_model_spec_for_modal() -> None:
    catalog = {item["id"]: item for item in workflow_catalog(PROJECT_ROOT)}

    item = catalog["comfy_workflow_source_path"]
    assert item["bindings"]
    assert item["model_count"] == 0
    assert item["size_gib"] == 0.0
    assert "model_ids" not in item


def test_selected_plan_requires_existing_bound_workflow(tmp_path: Path) -> None:
    project_root, user_root = _modal_test_project(tmp_path)
    workflow = user_root / "사용자가_수정한_이름.json"
    workflow.write_text('{"nodes": [], "links": []}\n', encoding="utf-8")
    plan = selected_install_plan(
        project_root,
        ["comfy_workflow_source_path"],
        {"comfy_workflow_source_path": str(workflow)},
    )

    assert plan["workflow_ids"] == ["comfy_workflow_source_path"]
    assert plan["model_count"] == 0
    assert plan["size_gib"] == 0.0
    assert plan["workflow_files"][0]["source_path"] == str(workflow.resolve())


def test_selected_plan_blocks_version_named_workflow_outside_soya_user(
    tmp_path: Path,
) -> None:
    project_root, _user_root = _modal_test_project(tmp_path)
    distribution = tmp_path / "배포_워크플로우__v1.json"
    distribution.write_text('{"nodes": [], "links": []}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="설치된 사용자 워크플로우가 아닙니다"):
        selected_install_plan(
            project_root,
            ["comfy_workflow_source_path"],
            {"comfy_workflow_source_path": str(distribution)},
        )


def test_selected_plan_reports_missing_workflow_file() -> None:
    with pytest.raises(FileNotFoundError, match="SOYA_USER 워크플로우 파일이 없습니다"):
        selected_install_plan(
            PROJECT_ROOT,
            ["comfy_workflow_source_path"],
            {"comfy_workflow_source_path": ""},
        )


def test_workflow_assets_follow_current_local_checkpoint_and_loras(tmp_path: Path) -> None:
    comfy_root = tmp_path / "comfy"
    checkpoint = comfy_root / "models" / "checkpoints" / "custom" / "changed.safetensors"
    power_lora = comfy_root / "models" / "loras" / "styles" / "power.safetensors"
    soya_lora = comfy_root / "models" / "loras" / "SOYA_CHAR_LORA" / "hero.safetensors"
    for path, content in (
        (checkpoint, b"checkpoint-from-user"),
        (power_lora, b"power-lora-from-user"),
        (soya_lora, b"soya-lora-from-user"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    workflow = {
        "nodes": [
            {"id": 1, "type": "CheckpointLoaderSimple", "widgets_values": ["custom/changed.safetensors"]},
            {
                "id": 2,
                "type": "Power Lora Loader (rgthree)",
                "widgets_values": [{"lora": "styles/power.safetensors", "strength": 0.8}],
            },
            {
                "id": 3,
                "type": "SoyaNode",
                "widgets_values": [
                    'prefix {"LORA_DATA":{"lora_name":"SOYA_CHAR_LORA/hero.safetensors"}} suffix'
                ],
            },
        ],
        "links": [],
    }

    assets = resolve_workflow_model_files(
        [workflow],
        build_local_model_index(comfy_root),
    )

    assert assets["model_count"] == 3
    assert assets["model_files"] == [
        {
            "source_path": str(checkpoint.resolve()),
            "remote_path": "checkpoints/custom/changed.safetensors",
            "size": checkpoint.stat().st_size,
            "sha256": assets["model_files"][0]["sha256"],
        }
    ]
    assert {item["source_path"] for item in assets["lora_files"]} == {
        str(power_lora.resolve()),
        str(soya_lora.resolve()),
    }
    assert {item["remote_path"] for item in assets["lora_files"]} == {
        "styles/power.safetensors",
        "SOYA_CHAR_LORA/hero.safetensors",
    }
    assert all(len(item["sha256"]) == 64 for item in assets["model_files"] + assets["lora_files"])


def test_modal_install_uploads_local_assets_without_remote_model_installer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow = tmp_path / "user-workflow.json"
    model = tmp_path / "user-checkpoint.safetensors"
    lora = tmp_path / "user-lora.safetensors"
    workflow.write_text('{"nodes": [], "links": []}', encoding="utf-8")
    model.write_bytes(b"local-checkpoint")
    lora.write_bytes(b"local-lora")

    class FakeBatch:
        def __init__(self, uploads: list[tuple[object, str]]) -> None:
            self.uploads = uploads

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def put_file(self, source, remote_path: str) -> None:
            self.uploads.append((source, remote_path))

    class FakeVolume:
        def __init__(self) -> None:
            self.uploads: list[tuple[object, str]] = []

        def read_file(self, _path: str):
            return [b"{}"]

        def batch_upload(self, *, force: bool):
            assert force is True
            return FakeBatch(self.uploads)

    volumes = {
        "test-workflows": FakeVolume(),
        "test-models": FakeVolume(),
        "test-loras": FakeVolume(),
    }
    monkeypatch.setattr(
        client_cli.modal.Volume,
        "from_name",
        lambda name, environment_name: volumes[name],
    )
    monkeypatch.setattr(
        client_cli.modal.Function,
        "from_name",
        lambda *_args, **_kwargs: pytest.fail("원격 install_models를 호출하면 안 됩니다."),
    )

    result = client_cli.install(
        {
            "app_name": "test",
            "environment": "main",
            "workflow_files": [
                {"source_path": str(workflow), "remote_name": "user-workflow.json"}
            ],
            "model_files": [
                {
                    "source_path": str(model),
                    "remote_path": "checkpoints/user-checkpoint.safetensors",
                    "size": model.stat().st_size,
                    "sha256": "a" * 64,
                }
            ],
            "lora_files": [
                {
                    "source_path": str(lora),
                    "remote_path": "user-lora.safetensors",
                    "size": lora.stat().st_size,
                    "sha256": "b" * 64,
                }
            ],
        }
    )

    assert result["uploaded_workflows"] == 1
    assert result["model_sync"] == {"uploaded": 1, "skipped": 0}
    assert result["lora_sync"] == {"uploaded": 1, "skipped": 0}
    assert (str(model), "/checkpoints/user-checkpoint.safetensors") in volumes[
        "test-models"
    ].uploads
    assert (str(lora), "/user-lora.safetensors") in volumes["test-loras"].uploads


@pytest.mark.asyncio
async def test_modal_service_install_plan_uses_current_soya_user_model_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root, user_root = _modal_test_project(tmp_path)
    checkpoint = project_root / "comfy" / "models" / "checkpoints" / "changed.safetensors"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"user-current-checkpoint")
    workflow = user_root / "내_워크플로우.json"
    workflow.write_text(
        json.dumps(
            {
                "1": {
                    "class_type": "CheckpointLoaderSimple",
                    "inputs": {"ckpt_name": "changed.safetensors"},
                }
            }
        ),
        encoding="utf-8",
    )
    config = {
        "modal_enabled": True,
        "comfy_workflow_source_path": str(workflow),
    }
    service = ModalService(project_root, lambda: config)
    observed: dict = {}

    async def connected(_settings: ModalSettings) -> bool:
        return True

    async def capture_commands(_args, **kwargs):
        payload = kwargs.get("stdin_payload")
        if isinstance(payload, dict):
            observed.update(payload)
            return 0, json.dumps({"ok": True, "result": {}}), ""
        return 0, "", ""

    monkeypatch.setattr(service, "account_connected", connected)
    monkeypatch.setattr(service, "_run_command", capture_commands)

    await service.start_install(["comfy_workflow_source_path"])
    assert service._install_task is not None
    await service._install_task

    assert service._install_state["state"] == "completed"
    assert observed["action"] == "install"
    assert observed["model_files"][0]["source_path"] == str(checkpoint.resolve())
    assert observed["model_files"][0]["remote_path"] == "checkpoints/changed.safetensors"
    assert observed["lora_files"] == []


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


def test_explicit_modal_input_folder_preserves_comfy_relative_paths(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    job_root = input_root / "modal_jobs" / "job-1"
    first = job_root / "1_train" / "face.png"
    second = job_root / "2_test" / "pose.png"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_bytes(b"face")
    second.write_bytes(b"pose")

    resolved = resolve_explicit_input_files(
        [job_root],
        {"comfy_input_dir": str(input_root)},
    )

    assert resolved == [
        {
            "source_path": str(first),
            "remote_name": "modal_jobs/job-1/1_train/face.png",
        },
        {
            "source_path": str(second),
            "remote_name": "modal_jobs/job-1/2_test/pose.png",
        },
    ]


def test_modal_lora_result_sync_is_non_destructive_and_uses_no_requirements_folder(
    tmp_path: Path,
) -> None:
    local_root = tmp_path / "installed-app" / "loras" / "SOYA_CHAR_LORA"
    service = ModalService(
        tmp_path / "installed-app",
        lambda: {"lora_load_path": str(local_root)},
    )
    remote_result = tmp_path / "download" / "hero.safetensors"
    remote_result.parent.mkdir(parents=True)
    remote_result.write_bytes(b"modal-v1")
    artifact = {
        "path": str(remote_result),
        "relative_path": "Alice/Lora/hero/hero.safetensors",
    }

    first = service._store_modal_artifacts([artifact], service.get_config())
    target = local_root / "Alice" / "Lora" / "hero" / "hero.safetensors"
    identical = service._store_modal_artifacts([artifact], service.get_config())
    remote_result.write_bytes(b"modal-v2")
    conflict = service._store_modal_artifacts([artifact], service.get_config())

    assert first[0]["status"] == "stored"
    assert identical[0]["status"] == "identical"
    assert conflict[0]["status"] == "conflict_copy"
    assert target.read_bytes() == b"modal-v1"
    assert Path(conflict[0]["local_path"]).read_bytes() == b"modal-v2"
    assert ".modal-" in Path(conflict[0]["local_path"]).name
    assert not (tmp_path / "installed-app" / "요구사항").exists()


@pytest.mark.asyncio
async def test_disabled_modal_does_not_create_delete_outbox(tmp_path: Path) -> None:
    service = ModalService(tmp_path, lambda: {"modal_enabled": False})

    await service.enqueue_lora_delete("SOYA_CHAR_LORA/Alice/Lora/hero")

    assert not (tmp_path / "modal_lora_delete_outbox.json").exists()


def test_modal_delete_outbox_uses_deployment_backup_directory(tmp_path: Path) -> None:
    service = ModalService(tmp_path, lambda: {"modal_enabled": False})
    first = [{"remote_path": "Alice/Lora/hero"}]
    second = [{"remote_path": "Bob/Lora/hero"}]

    service._save_delete_outbox(first)
    service._save_delete_outbox(second)

    backups = list(
        (tmp_path / "backups" / "modal").glob(
            "modal_lora_delete_outbox_before_save_*.json"
        )
    )
    assert len(backups) == 1
    assert json.loads(backups[0].read_text(encoding="utf-8")) == first
    assert not (tmp_path / "요구사항").exists()


@pytest.mark.asyncio
async def test_disabled_modal_status_checks_account_but_skips_billing_and_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = ModalService(tmp_path, lambda: {"modal_enabled": False})

    async def connected_account_check(_settings: ModalSettings) -> bool:
        return True

    async def unexpected_billing(*_args, **_kwargs) -> dict:
        raise AssertionError("Modal OFF 상태에서는 청구 API를 실행하면 안 됩니다.")

    monkeypatch.setattr(service, "account_connected", connected_account_check)
    monkeypatch.setattr(service, "_billing_for_settings", unexpected_billing)

    status = await service.status(include_runtime=True)

    assert status["connected"] is True
    assert status["connection_checked"] is True
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
        "comfy_task_allocations": {"illustration": "modal"},
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
        "comfy_task_allocations": {"illustration": "modal"},
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
async def test_local_and_modal_lanes_claim_parallel_queue_items_without_duplicates() -> None:
    manager = QueueManager()
    manager.get_config = lambda: {
        "modal_enabled": True,
        "modal_max_concurrency": 1,
        "illustration_provider": "comfy",
        "bot_selected": "test-bot",
        "comfy_task_allocations": {"illustration": 1},
        "comfy_task_modal_parallel": {"illustration": True},
    }
    both_started = asyncio.Event()
    release = asyncio.Event()
    starts = []

    async def execute(item):
        starts.append((item.id, item.comfy_execution_target))
        if len(starts) == 2:
            both_started.set()
        await release.wait()
        return {"ok": True}

    async def no_prune(_item):
        return None

    manager._execute_item = execute
    manager._deferred_prune = no_prune
    first = await manager.add_item("illustration", "first", {}, priority=0)
    second = await manager.add_item("illustration", "second", {}, priority=0)

    try:
        await asyncio.wait_for(both_started.wait(), timeout=1)
        assert {item_id for item_id, _target in starts} == {first.id, second.id}
        assert {target for _item_id, target in starts} == {"local", "modal"}

        release.set()
        await asyncio.wait_for(
            asyncio.gather(first.completion_future, second.completion_future),
            timeout=1,
        )
        assert first.status == second.status == "completed"
    finally:
        release.set()
        tasks = [
            task for task in manager._modal_worker_tasks.values() if not task.done()
        ]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_instance_analysis_forks_can_run_on_local_and_modal_concurrently() -> None:
    tool = AssetToolMode()
    local_tool = tool.fork_for_execution()
    modal_tool = tool.fork_for_execution()
    both_started = asyncio.Event()
    release = asyncio.Event()
    active = 0
    peak = 0

    async def analyze(*_args, **_kwargs):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        if active == 2:
            both_started.set()
        await release.wait()
        active -= 1
        return {"success": True, "tags": ["tag"]}

    local_tool._analyze_internal = analyze
    modal_tool._analyze_internal = analyze
    local_task = asyncio.create_task(local_tool.analyze_image(b"local"))
    modal_task = asyncio.create_task(modal_tool.analyze_image(b"modal"))

    try:
        await asyncio.wait_for(both_started.wait(), timeout=1)
        release.set()
        results = await asyncio.wait_for(
            asyncio.gather(local_task, modal_task),
            timeout=1,
        )
        assert peak == 2
        assert all(result["success"] for result in results)
    finally:
        release.set()
        await asyncio.gather(local_task, modal_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_managed_workflow_run_tracks_result_without_local_comfy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root, user_root = _modal_test_project(tmp_path)
    workflow = user_root / "workflow-api.json"
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
    service = ModalService(project_root, lambda: config)

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
        assert payload["model_files"] == []
        assert payload["lora_files"] == []
        return converted

    async def generated(actual_workflow: dict, *, timeout_seconds: int = 3300):
        assert timeout_seconds == 3300
        assert actual_workflow == converted
        return b"png", {
            "prompt_id": "prompt-1",
            "content_type": "image/png",
            "model_sync": {"uploaded": 0},
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
    assert completed["model_sync"] == {"uploaded": 0}
    assert "image_bytes" not in completed
    assert image == b"png"
    assert content_type == "image/png"
