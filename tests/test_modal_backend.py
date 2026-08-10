from __future__ import annotations

import asyncio
import builtins
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
import urllib.error

import pytest

from modal_backend import client_cli
from modal_backend.manifest import (
    list_soya_user_workflows,
    plan_from_soya_user_names,
    selected_install_plan,
    workflow_catalog,
)
from modal_backend.service import (
    INSTALL_PROGRESS_PREFIX,
    ModalClientActionError,
    ModalService,
    WebStartCancelled,
    cost_summary,
)
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


def test_modal_manifest_import_does_not_require_comfy_installer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_path = PROJECT_ROOT / "modal_backend" / "manifest.py"
    spec = importlib.util.spec_from_file_location(
        "_modal_manifest_without_comfy_installer",
        module_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    real_import = builtins.__import__

    def import_without_comfy_installer(name: str, *args, **kwargs):
        if name == "comfy_installer" or name.startswith("comfy_installer."):
            raise ModuleNotFoundError("blocked to emulate the Modal runtime")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_comfy_installer)
    spec.loader.exec_module(module)

    assert callable(module.load_manifest)


def test_modal_defaults_are_scale_to_zero_and_independently_gpu_budgeted() -> None:
    settings = ModalSettings.from_mapping({})
    cost = cost_summary(settings)

    assert settings.gpu == "L4"
    assert settings.worker_gpu == "L4"
    assert settings.web_gpu == "L4"
    assert settings.max_concurrency == 2
    assert settings.scaledown_window_seconds == 15
    assert settings.status_refresh_seconds == 5
    assert settings.container_start_max_retries == 2
    assert settings.web_fast is False
    assert settings.public_dict()["container_start_max_retries"] == 2
    assert settings.public_dict()["web_fast"] is False
    assert cost["assumptions"]["min_containers"] == 0
    assert cost["assumptions"]["scaledown_window_seconds"] == 15
    assert cost["worker"]["gpu_per_hour"] == pytest.approx(0.7992)
    assert cost["web"]["gpu_per_hour"] == pytest.approx(0.7992)
    assert cost["combined_container_per_hour"] == pytest.approx(2.2314)
    assert cost["estimated_container_hours"] == pytest.approx(26.89)

    split = ModalSettings.from_mapping(
        {"modal_worker_gpu": "L40S", "modal_web_gpu": "RTX-PRO-6000"}
    )
    split_cost = cost_summary(split)
    assert split.worker_gpu == "L40S"
    assert split.web_gpu == "RTX-PRO-6000"
    assert split_cost["worker"]["gpu_per_hour"] == pytest.approx(1.9512)
    assert split_cost["web"]["gpu_per_hour"] == pytest.approx(3.0312)
    assert {
        profile["id"] for profile in split.public_dict()["gpu_profiles"]
    } == {"L4", "L40S", "RTX-PRO-6000"}


@pytest.mark.parametrize(
    "config",
    [
        {"modal_gpu": "T4"},
        {"modal_worker_gpu": "T4"},
        {"modal_web_gpu": "T4"},
        {"modal_worker_gpu": "A10"},
        {"modal_worker_gpu": "A100-40GB"},
        {"modal_worker_gpu": "A100-80GB"},
        {"modal_worker_gpu": "H100"},
        {"modal_max_concurrency": 0},
        {"modal_max_concurrency": 11},
        {"modal_profile": "bad profile"},
        {"modal_environment": "a" * 64},
        {"modal_scaledown_window_seconds": 1},
        {"modal_scaledown_window_seconds": 1201},
        {"modal_status_refresh_seconds": 1},
        {"modal_status_refresh_seconds": 61},
        {"modal_container_start_max_retries": -1},
        {"modal_container_start_max_retries": 11},
        {"modal_container_start_max_retries": True},
        {"modal_web_fast": "true"},
    ],
)
def test_modal_settings_reject_invalid_values(config: dict) -> None:
    with pytest.raises(ValueError):
        ModalSettings.from_mapping(config)


def test_modal_runtime_stats_uses_cls_method_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict = {}

    class FakeStats:
        backlog = 3
        num_total_runners = 2
        num_running_inputs = 1
        input_headroom = 4

    class FakeGenerate:
        @staticmethod
        def get_current_stats() -> FakeStats:
            return FakeStats()

    class FakeWorker:
        generate = FakeGenerate()

    class FakeCls:
        def with_options(self, **options) -> "FakeCls":
            observed["dynamic_options"] = options
            return self

        def __call__(self) -> FakeWorker:
            return FakeWorker()

    def from_name(
        app_name: str,
        class_name: str,
        *,
        environment_name: str,
    ) -> FakeCls:
        observed.update(
            {
                "app_name": app_name,
                "class_name": class_name,
                "environment_name": environment_name,
            }
        )
        return FakeCls()

    monkeypatch.setattr(client_cli.modal.Cls, "from_name", from_name)
    monkeypatch.setattr(
        client_cli.modal.Function,
        "from_name",
        lambda *_args, **_kwargs: pytest.fail(
            "@app.cls 메서드 통계 조회에 Function.from_name을 사용하면 안 됩니다."
        ),
    )

    result = client_cli.runtime_stats(
        {
            "app_name": "test-app",
            "environment": "main",
            "worker_gpu": "L40S",
        }
    )

    assert observed == {
        "app_name": "test-app",
        "class_name": "ComfyWorker",
        "environment_name": "main",
        "dynamic_options": {"gpu": "L40S"},
    }
    assert result == {
        "backlog": 3,
        "num_total_runners": 2,
        "num_running_inputs": 1,
        "input_headroom": 4,
    }


def test_modal_dynamic_worker_function_applies_selected_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict = {}

    class FakeFunction:
        def with_options(self, **options) -> "FakeFunction":
            observed["dynamic_options"] = options
            return self

    def from_name(
        app_name: str,
        function_name: str,
        *,
        environment_name: str,
    ) -> FakeFunction:
        observed.update(
            app_name=app_name,
            function_name=function_name,
            environment_name=environment_name,
        )
        return FakeFunction()

    monkeypatch.setattr(client_cli.modal.Function, "from_name", from_name)

    result = client_cli._dynamic_worker_function(
        {
            "app_name": "test-app",
            "environment": "main",
            "worker_gpu": "RTX-PRO-6000",
        },
        "gpu_probe",
    )

    assert isinstance(result, FakeFunction)
    assert observed == {
        "app_name": "test-app",
        "function_name": "gpu_probe",
        "environment_name": "main",
        "dynamic_options": {"gpu": "RTX-PRO-6000"},
    }


def test_modal_client_error_reason_uses_sdk_exception_types() -> None:
    assert (
        client_cli._error_reason(client_cli.modal.exception.NotFoundError("missing"))
        == "app_not_deployed"
    )
    assert (
        client_cli._error_reason(client_cli.modal.exception.ConnectionError("offline"))
        == "network_unavailable"
    )
    assert (
        client_cli._error_reason(client_cli.modal.exception.ServiceError("unavailable"))
        == "network_unavailable"
    )
    assert client_cli._error_reason(ValueError("bad data")) == "runtime_unavailable"


def test_modal_call_start_retry_limit_cancels_function_and_container(
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FakeLogs:
        def __init__(self, call) -> None:
            self.call = call

        def tail(self, *, entries: int):
            assert entries == client_cli.CALL_START_LOG_TAIL_ENTRIES
            return [
                SimpleNamespace(
                    message="container boot",
                    context_ids=["in-1", f"ta-{index}"],
                )
                for index in range(1, self.call.poll_count + 1)
            ]

    class FakeCall:
        def __init__(self) -> None:
            self.poll_count = 0
            self.logs = FakeLogs(self)
            self.cancel_requests: list[bool] = []

        def get(self, *, timeout: float):
            assert timeout > 0
            self.poll_count += 1
            raise client_cli.modal.exception.TimeoutError("still starting")

        def cancel(self, *, terminate_containers: bool) -> None:
            self.cancel_requests.append(terminate_containers)

    call = FakeCall()

    with pytest.raises(
        client_cli.ModalContainerStartRetryLimitError,
        match="추가 재시도 2회",
    ):
        client_cli._wait_for_call_with_start_retry_limit(
            call,
            timeout_seconds=30,
            max_retries=2,
            operation="generate",
        )

    assert call.poll_count == 4
    assert call.cancel_requests == [True]
    captured = capsys.readouterr()
    assert "attempt=4/3" in captured.err
    assert "원격 호출과 실행 컨테이너를 취소" in captured.err


def test_modal_call_start_monitor_stops_after_remote_method_marker(
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FakeLogs:
        def __init__(self, call) -> None:
            self.call = call

        def tail(self, *, entries: int):
            assert entries == client_cli.CALL_START_LOG_TAIL_ENTRIES
            result = [
                SimpleNamespace(
                    message="container boot",
                    context_ids=["in-1", "ta-1"],
                )
            ]
            if self.call.poll_count >= 2:
                result.append(
                    SimpleNamespace(
                        message=(
                            client_cli.CALL_STARTED_LOG_PREFIX
                            + '{"operation":"generate"}'
                        ),
                        context_ids=["in-1", "ta-1"],
                    )
                )
            return result

    class FakeCall:
        def __init__(self) -> None:
            self.poll_count = 0
            self.logs = FakeLogs(self)
            self.cancel_requests: list[bool] = []

        def get(self, *, timeout: float):
            assert timeout > 0
            self.poll_count += 1
            if self.poll_count <= 2:
                raise client_cli.modal.exception.TimeoutError("still starting")
            return {"images": []}

        def cancel(self, *, terminate_containers: bool) -> None:
            self.cancel_requests.append(terminate_containers)

    call = FakeCall()

    result = client_cli._wait_for_call_with_start_retry_limit(
        call,
        timeout_seconds=30,
        max_retries=2,
        operation="generate",
    )

    assert result == {"images": []}
    assert call.poll_count == 3
    assert call.cancel_requests == []
    assert "원격 메서드 진입 확인" in capsys.readouterr().err


def test_modal_call_start_monitor_retries_builtin_poll_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCall:
        def __init__(self) -> None:
            self.poll_count = 0

        def get(self, *, timeout: float):
            assert timeout > 0
            self.poll_count += 1
            if self.poll_count == 1:
                raise TimeoutError("still running")
            return {"images": []}

    call = FakeCall()
    monkeypatch.setattr(
        client_cli,
        "_call_start_observations",
        lambda _call: (True, {"ta-1"}),
    )

    result = client_cli._wait_for_call_with_start_retry_limit(
        call,
        timeout_seconds=30,
        max_retries=2,
        operation="generate",
    )

    assert result == {"images": []}
    assert call.poll_count == 2


def test_modal_client_main_serializes_app_not_deployed_reason(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def missing_runtime(_payload: dict) -> dict:
        raise client_cli.modal.exception.NotFoundError("missing app")

    monkeypatch.setattr(
        client_cli,
        "_read_payload",
        lambda: {"action": "runtime_stats"},
    )
    monkeypatch.setattr(client_cli, "runtime_stats", missing_runtime)

    assert client_cli.main() == 1
    captured = capsys.readouterr()
    response = json.loads(captured.out)
    assert response == {
        "ok": False,
        "reason": "app_not_deployed",
        "error_type": "NotFoundError",
        "error": "NotFoundError: missing app",
    }
    assert "Traceback (most recent call last)" in captured.err


@pytest.mark.asyncio
async def test_modal_client_action_preserves_structured_failure_reason(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = ModalService(tmp_path, lambda: {"modal_enabled": True})

    async def failed_command(*_args, **_kwargs) -> tuple[int, str, str]:
        return (
            1,
            json.dumps(
                {
                    "ok": False,
                    "reason": "app_not_deployed",
                    "error_type": "NotFoundError",
                    "error": "NotFoundError: missing",
                }
            ),
            "remote traceback",
        )

    monkeypatch.setattr(service, "_run_command", failed_command)

    with pytest.raises(ModalClientActionError) as caught:
        await service._run_client_action(
            ModalSettings.from_mapping({}),
            "runtime_stats",
            timeout=30,
        )

    assert caught.value.reason == "app_not_deployed"
    assert caught.value.error_type == "NotFoundError"


@pytest.mark.asyncio
async def test_modal_client_action_passes_container_start_retry_setting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = ModalService(tmp_path, lambda: {})
    observed: dict = {}

    async def successful_command(*_args, **kwargs) -> tuple[int, str, str]:
        observed.update(kwargs["stdin_payload"])
        return 0, json.dumps({"ok": True, "result": {}}), ""

    monkeypatch.setattr(service, "_run_command", successful_command)
    settings = ModalSettings.from_mapping(
        {"modal_container_start_max_retries": 4}
    )

    await service._run_client_action(
        settings,
        "runtime_stats",
        timeout=30,
    )

    assert observed["container_start_max_retries"] == 4
    assert observed["worker_gpu"] == "L4"


@pytest.mark.asyncio
async def test_modal_run_workflow_preserves_start_retry_limit_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = ModalService(
        tmp_path,
        lambda: {
            "modal_enabled": True,
            "modal_worker_gpu": "RTX-PRO-6000",
            "modal_container_start_max_retries": 2,
        },
    )
    observed: dict = {}

    async def connected(_settings: ModalSettings) -> bool:
        return True

    async def assets(_workflows: list[dict]) -> dict:
        return {"model_files": [], "lora_files": []}

    async def failed_command(*_args, **kwargs) -> tuple[int, str, str]:
        observed.update(kwargs["stdin_payload"])
        return (
            1,
            json.dumps(
                {
                    "ok": False,
                    "error": (
                        "Modal generate 컨테이너 시작이 최초 1회와 "
                        "추가 재시도 2회를 초과해 취소되었습니다."
                    ),
                }
            ),
            "retry limit exceeded",
        )

    monkeypatch.setattr(service, "account_connected", connected)
    monkeypatch.setattr(service, "_resolve_local_workflow_assets", assets)
    monkeypatch.setattr(service, "_run_command", failed_command)

    with pytest.raises(RuntimeError, match="추가 재시도 2회"):
        await service.run_workflow(
            {"1": {"class_type": "EmptyLatentImage", "inputs": {}}},
        )

    assert observed["container_start_max_retries"] == 2
    assert observed["worker_gpu"] == "RTX-PRO-6000"


@pytest.mark.asyncio
async def test_modal_run_workflow_includes_soya_cache_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_root = tmp_path / "input"
    character_root = input_root / "soya_bot" / "sample_bot" / "alice"
    character_root.mkdir(parents=True)
    cache_pt = character_root / "cache.pt"
    cache_ipadapter = character_root / "cache.ipadpt"
    cache_pt.write_bytes(b"embedding-cache")
    cache_ipadapter.write_bytes(b"ipadapter-cache")
    positive = "\n".join(
        [
            "[CACHE_PATH]",
            json.dumps(
                {"list": [{"emb_path": "soya_bot/sample_bot/alice/cache.pt"}]}
            ),
            "[FACE_ID_DIR]",
            json.dumps(
                {"list": [{"ipa_path": "soya_bot/sample_bot/alice/cache.ipadpt"}]}
            ),
        ]
    )
    workflow = {
        "9": {"class_type": "PrimitiveStringMultiline", "inputs": {"value": positive}},
        "909": {"class_type": "SoyaPromptParser_mdsoya", "inputs": {"text": ["9", 0]}},
        "458": {
            "class_type": "SoyaIPAPatchMaker_mdsoya",
            "inputs": {
                "embed_cache_data": ["909", 8],
                "ipa_cache_data": ["909", 10],
            },
        },
    }
    service = ModalService(
        tmp_path,
        lambda: {"modal_enabled": True, "comfy_input_dir": str(input_root)},
    )
    observed: dict = {}

    async def connected(_settings: ModalSettings) -> bool:
        return True

    async def assets(_workflows: list[dict]) -> dict:
        return {"model_files": [], "lora_files": []}

    async def failed_command(*_args, **kwargs) -> tuple[int, str, str]:
        observed.update(kwargs["stdin_payload"])
        return 1, json.dumps({"ok": False, "error": "test stop"}), ""

    monkeypatch.setattr(service, "account_connected", connected)
    monkeypatch.setattr(service, "_resolve_local_workflow_assets", assets)
    monkeypatch.setattr(service, "_run_command", failed_command)

    with pytest.raises(RuntimeError, match="test stop"):
        await service.run_workflow(workflow)

    assert observed["input_files"] == [
        {
            "source_path": str(cache_pt),
            "remote_name": "soya_bot/sample_bot/alice/cache.pt",
        },
        {
            "source_path": str(cache_ipadapter),
            "remote_name": "soya_bot/sample_bot/alice/cache.ipadpt",
        },
    ]


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


def test_list_soya_user_workflows_enumerates_files_by_name(tmp_path: Path) -> None:
    project_root, user_root = _modal_test_project(tmp_path)
    (user_root / "beta.json").write_text("{}", encoding="utf-8")
    (user_root / "alpha.json").write_text("{}", encoding="utf-8")
    (user_root / "not_json.txt").write_text("x", encoding="utf-8")

    entries = list_soya_user_workflows(project_root)
    names = [entry["name"] for entry in entries]
    assert names == ["alpha.json", "beta.json"]
    for entry in entries:
        assert entry["source_path"].endswith(entry["name"])
        # resolve된 경로는 SOYA_USER 하위
        assert Path(entry["source_path"]).resolve().relative_to(user_root.resolve())


def test_list_soya_user_workflows_empty_when_missing(tmp_path: Path) -> None:
    project_root, _user_root = _modal_test_project(tmp_path)
    assert list_soya_user_workflows(project_root) == []


def test_plan_from_soya_user_names_uses_filename_as_id(tmp_path: Path) -> None:
    project_root, user_root = _modal_test_project(tmp_path)
    (user_root / "foo.json").write_text('{"nodes": []}', encoding="utf-8")

    plan = plan_from_soya_user_names(project_root, ["foo.json"])
    assert plan["workflow_ids"] == ["foo.json"]
    assert plan["workflow_files"][0]["id"] == "foo.json"
    assert plan["workflow_files"][0]["remote_name"] == "foo.json"
    assert plan["workflow_files"][0]["source_path"].endswith("foo.json")


def test_plan_from_soya_user_names_rejects_path_traversal(tmp_path: Path) -> None:
    project_root, user_root = _modal_test_project(tmp_path)
    (user_root / "inside.json").write_text("{}", encoding="utf-8")
    # workflows 루트(=SOYA_USER 밖)에 파일을 두어도 파일명만 받으므로 접근 불가
    outside = user_root.parent / "outside.json"
    outside.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="파일명만"):
        plan_from_soya_user_names(project_root, ["../outside.json"])
    with pytest.raises(ValueError, match="파일명만"):
        plan_from_soya_user_names(project_root, [str(outside)])
    with pytest.raises(ValueError, match="파일명만"):
        plan_from_soya_user_names(project_root, ["sub/inside.json"])
    # 존재하지 않는 파일명 → FileNotFoundError(파일명 자체는 합법)
    with pytest.raises(FileNotFoundError):
        plan_from_soya_user_names(project_root, ["nope.json"])
    with pytest.raises(ValueError, match="하나 이상"):
        plan_from_soya_user_names(project_root, [])


def test_plan_from_soya_user_names_blocks_symlink_escape(tmp_path: Path) -> None:
    """SOYA_USER 안의 심볼릭 링크가 밖을 가리키면 거부된다.

    _require_user_workflow의 Path.resolve()가 심볼릭 링크 대상을 따라가므로
    resolved 경로가 user_root 밖이 되어 relative_to 검증에 걸린다.
    Windows에서 심볼릭 링크 생성에 관리자 권한이 필요하면 생성 단계에서 skip.
    """
    import os

    project_root, user_root = _modal_test_project(tmp_path)
    target = user_root.parent / "escaped.json"
    target.write_text("{}", encoding="utf-8")
    link = user_root / "link.json"
    try:
        os.symlink(target, link)
    except OSError as exc:
        pytest.skip(f"심볼릭 링크 생성 불가(권한): {exc}")

    with pytest.raises(ValueError, match="설치된 사용자 워크플로우가 아닙니다"):
        plan_from_soya_user_names(project_root, ["link.json"])


def test_modal_service_workflows_lists_soya_user_files_configured_true(
    tmp_path: Path,
) -> None:
    """service.workflows()는 SOYA_USER 파일 나열이며 모두 configured=True다.

    configured=False는 'SOYA_USER 밖' 거부가 아니라 이 카탈로그에서는 의미가 없다.
    이 계약: 성공적으로 나열된 항목은 항상 configured=True. 파싱 실패 파일은
    목록에서 제외되고 errors로만 보고된다.
    """
    project_root, user_root = _modal_test_project(tmp_path)
    checkpoint = project_root / "comfy" / "models" / "checkpoints" / "m.safetensors"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"ckpt")
    (user_root / "good.json").write_text(
        json.dumps({"1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "m.safetensors"}}}),
        encoding="utf-8",
    )

    service = ModalService(project_root, lambda: {})
    payload = service.workflows()

    assert isinstance(payload, dict)
    assert [w["id"] for w in payload["workflows"]] == ["good.json"]
    good = payload["workflows"][0]
    assert good["configured"] is True
    assert good["source_name"] == "good.json"
    assert good["model_count"] == 1
    assert payload["errors"] == []


def test_modal_service_workflows_excludes_broken_json_per_file(
    tmp_path: Path,
) -> None:
    """한 파일의 깨진 JSON이 전체 /workflows 응답을 터뜨리지 않는다.

    깨진 파일은 workflows 목록에서 제외되고 errors에 이름/사유로 보고된다.
    정상 파일은 정상적으로 나열된다.
    """
    project_root, user_root = _modal_test_project(tmp_path)
    (user_root / "good.json").write_text('{"1": {"class_type": "EmptyLatentImage"}}', encoding="utf-8")
    (user_root / "broken.json").write_text("{not json", encoding="utf-8")

    service = ModalService(project_root, lambda: {})
    payload = service.workflows()

    assert [w["id"] for w in payload["workflows"]] == ["good.json"]
    assert all(w["configured"] is True for w in payload["workflows"])
    error_names = [e["name"] for e in payload["errors"]]
    assert error_names == ["broken.json"]
    assert payload["errors"][0]["error"]  # 사유 문자열 존재


def test_missing_sync_manifest_is_normal_first_sync_state(
    capsys: pytest.CaptureFixture[str],
) -> None:
    class MissingManifestVolume:
        def read_file(self, _path: str):
            raise FileNotFoundError("manifest does not exist")

    assert client_cli._read_sync_manifest(
        MissingManifestVolume(),
        client_cli.MODEL_SYNC_MANIFEST_PATH,
        "모델",
    ) == {}
    assert "동기화 명세 읽기 실패" not in capsys.readouterr().err


def test_modal_client_lists_and_hashes_only_root_json_workflows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    good = b'{"1":{"class_type":"EmptyLatentImage"}}'
    broken = b"{not json"

    class FakeVolume:
        def listdir(self, path: str, *, recursive: bool):
            assert path == "/"
            assert recursive is False
            file_type = SimpleNamespace(name="FILE")
            directory_type = SimpleNamespace(name="DIRECTORY")
            return [
                SimpleNamespace(path="/good.json", type=file_type, size=len(good), mtime=10),
                SimpleNamespace(path="/broken.json", type=file_type, size=len(broken), mtime=20),
                SimpleNamespace(path="/nested", type=directory_type, size=0, mtime=0),
            ]

        def read_file(self, path: str):
            if path == "/good.json":
                return [good]
            if path == "/broken.json":
                return [broken]
            raise AssertionError(path)

    monkeypatch.setattr(
        client_cli.modal.Volume,
        "from_name",
        lambda name, environment_name: FakeVolume(),
    )

    result = client_cli.list_workflows(
        {"app_name": "test", "environment": "main"}
    )

    assert [item["name"] for item in result["workflows"]] == [
        "broken.json",
        "good.json",
    ]
    good_result = next(item for item in result["workflows"] if item["name"] == "good.json")
    assert good_result["valid"] is True
    assert good_result["sha256"] == hashlib.sha256(good).hexdigest()
    broken_result = next(
        item for item in result["workflows"] if item["name"] == "broken.json"
    )
    assert broken_result["valid"] is False
    assert [item["name"] for item in result["errors"]] == ["broken.json"]


@pytest.mark.asyncio
async def test_remote_workflow_query_compares_local_and_remote_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root, user_root = _modal_test_project(tmp_path)
    same = user_root / "same.json"
    changed = user_root / "changed.json"
    missing = user_root / "missing.json"
    same.write_text('{"1":{"class_type":"Same"}}', encoding="utf-8")
    changed.write_text('{"1":{"class_type":"Local"}}', encoding="utf-8")
    missing.write_text('{"1":{"class_type":"Missing"}}', encoding="utf-8")
    service = ModalService(project_root, lambda: {"modal_enabled": True})

    async def connected(_settings: ModalSettings) -> bool:
        return True

    async def client_action(
        _settings: ModalSettings,
        action: str,
        *,
        timeout: float,
        **_payload,
    ) -> dict:
        assert action == "list_workflows"
        assert timeout == 120
        return {
            "workflows": [
                {
                    "name": "same.json",
                    "sha256": service._sha256_file(same),
                    "size": same.stat().st_size,
                    "mtime": 1,
                    "valid": True,
                },
                {
                    "name": "changed.json",
                    "sha256": "f" * 64,
                    "size": 10,
                    "mtime": 2,
                    "valid": True,
                },
                {
                    "name": "remote-only.json",
                    "sha256": "e" * 64,
                    "size": 10,
                    "mtime": 3,
                    "valid": True,
                },
            ],
            "errors": [],
        }

    monkeypatch.setattr(service, "account_connected", connected)
    monkeypatch.setattr(service, "_run_client_action", client_action)

    result = await service.remote_workflows()
    states = {item["id"]: item["sync_state"] for item in result["workflows"]}

    assert states == {
        "changed.json": "different",
        "missing.json": "missing",
        "same.json": "synced",
    }
    assert "remote-only.json" not in states
    assert result["counts"] == {
        "synced": 1,
        "different": 1,
        "missing": 1,
        "invalid": 0,
    }


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
    capsys: pytest.CaptureFixture[str],
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
    volume_requests: list[tuple[str, str, bool]] = []

    def volume_from_name(
        name: str,
        *,
        environment_name: str,
        create_if_missing: bool = False,
    ) -> FakeVolume:
        volume_requests.append((name, environment_name, create_if_missing))
        return volumes[name]

    monkeypatch.setattr(
        client_cli.modal.Volume,
        "from_name",
        volume_from_name,
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
    assert volume_requests == [
        ("test-workflows", "main", True),
        ("test-models", "main", True),
        ("test-loras", "main", True),
    ]
    progress_lines = [
        line
        for line in capsys.readouterr().err.splitlines()
        if line.startswith(client_cli.INSTALL_PROGRESS_PREFIX)
    ]
    progress_events = [
        json.loads(line[len(client_cli.INSTALL_PROGRESS_PREFIX) :])
        for line in progress_lines
    ]
    assert [event["label"] for event in progress_events if event["event"] == "batch_complete"] == [
        "워크플로우",
        "모델",
        "LoRA",
    ]


def test_modal_install_replaces_workflow_without_reuploading_matching_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow = tmp_path / "changed.json"
    model = tmp_path / "same.safetensors"
    workflow.write_text('{"1":{"class_type":"Changed"}}', encoding="utf-8")
    model.write_bytes(b"same-model")
    expected = {"sha256": "a" * 64, "size": model.stat().st_size}

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
        def __init__(self, manifest: dict | None = None) -> None:
            self.manifest = manifest or {}
            self.uploads: list[tuple[object, str]] = []

        def read_file(self, _path: str):
            return [json.dumps(self.manifest).encode("utf-8")]

        def batch_upload(self, *, force: bool):
            assert force is True
            return FakeBatch(self.uploads)

    volumes = {
        "test-workflows": FakeVolume(),
        "test-models": FakeVolume({"checkpoints/same.safetensors": expected}),
        "test-loras": FakeVolume(),
    }
    monkeypatch.setattr(
        client_cli.modal.Volume,
        "from_name",
        lambda name, *, environment_name, create_if_missing=False: volumes[name],
    )

    result = client_cli.install(
        {
            "app_name": "test",
            "environment": "main",
            "workflow_files": [
                {"source_path": str(workflow), "remote_name": workflow.name}
            ],
            "model_files": [
                {
                    "source_path": str(model),
                    "remote_path": "checkpoints/same.safetensors",
                    **expected,
                }
            ],
            "lora_files": [],
        }
    )

    assert result["model_sync"] == {"uploaded": 0, "skipped": 1}
    assert volumes["test-models"].uploads == []
    assert (str(workflow), f"/{workflow.name}") in volumes["test-workflows"].uploads


@pytest.mark.asyncio
async def test_modal_streaming_command_forwards_stdout_and_stderr() -> None:
    observed: list[tuple[str, str]] = []

    code, stdout, stderr = await ModalService._run_command(
        [
            sys.executable,
            "-c",
            "import sys; print('✓ build-line', flush=True); "
            "print('✓ warning-line', file=sys.stderr, flush=True)",
        ],
        env=ModalService._subprocess_env("test-profile"),
        timeout=30,
        output_callback=lambda source, line: observed.append((source, line)),
    )

    assert code == 0
    assert stdout == "✓ build-line\n"
    assert stderr == "✓ warning-line\n"
    assert ("stdout", "✓ build-line") in observed
    assert ("stderr", "✓ warning-line") in observed


@pytest.mark.asyncio
async def test_modal_nonstreaming_command_avoids_default_executor_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def unexpected_to_thread(*_args, **_kwargs):
        raise AssertionError("Modal command waits must not use a default executor thread")

    monkeypatch.setattr(asyncio, "to_thread", unexpected_to_thread)

    code, stdout, stderr = await ModalService._run_command(
        [sys.executable, "-c", "print('async-subprocess')"],
        env=dict(os.environ),
        timeout=30,
    )

    assert code == 0
    assert stdout == "async-subprocess\n"
    assert stderr == ""


@pytest.mark.asyncio
async def test_modal_status_action_single_flight_deduplicates_identical_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = ModalService(tmp_path, lambda: {})
    settings = ModalSettings.from_mapping({})
    started = asyncio.Event()
    release = asyncio.Event()
    call_count = 0

    async def run_once(
        _settings: ModalSettings,
        action: str,
        *,
        timeout: float,
        **_payload,
    ) -> dict:
        nonlocal call_count
        assert action == "runtime_stats"
        assert timeout == 30
        call_count += 1
        started.set()
        await release.wait()
        return {"num_total_runners": 1}

    monkeypatch.setattr(service, "_run_client_action_once", run_once)

    first = asyncio.create_task(
        service._run_client_action(settings, "runtime_stats", timeout=30)
    )
    await asyncio.wait_for(started.wait(), timeout=1)
    second = asyncio.create_task(
        service._run_client_action(settings, "runtime_stats", timeout=30)
    )
    await asyncio.sleep(0)

    assert call_count == 1
    release.set()
    assert await asyncio.gather(first, second) == [
        {"num_total_runners": 1},
        {"num_total_runners": 1},
    ]


@pytest.mark.asyncio
async def test_modal_account_check_single_flight_deduplicates_profile_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = ModalService(tmp_path, lambda: {})
    settings = ModalSettings.from_mapping({"modal_profile": "shared-profile"})
    started = asyncio.Event()
    release = asyncio.Event()
    call_count = 0

    async def connected_once(_settings: ModalSettings) -> bool:
        nonlocal call_count
        call_count += 1
        started.set()
        await release.wait()
        return True

    monkeypatch.setattr(service, "_account_connected_once", connected_once)

    first = asyncio.create_task(service.account_connected(settings))
    await asyncio.wait_for(started.wait(), timeout=1)
    second = asyncio.create_task(service.account_connected(settings))
    await asyncio.sleep(0)

    assert call_count == 1
    release.set()
    assert await asyncio.gather(first, second) == [True, True]


@pytest.mark.asyncio
async def test_modal_worker_and_web_status_queries_start_concurrently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = ModalService(tmp_path, lambda: {})
    started: set[str] = set()
    both_started = asyncio.Event()
    release = asyncio.Event()

    async def wait_for_peer(name: str) -> dict:
        started.add(name)
        if len(started) == 2:
            both_started.set()
        await release.wait()
        return {"state": "stopped", "source": name}

    monkeypatch.setattr(
        service,
        "_worker_status_block",
        lambda _settings: wait_for_peer("worker"),
    )
    monkeypatch.setattr(
        service,
        "_dock_web_status",
        lambda _settings: wait_for_peer("web"),
    )

    status_task = asyncio.create_task(service.worker_status())
    await asyncio.wait_for(both_started.wait(), timeout=1)
    assert started == {"worker", "web"}

    release.set()
    status = await status_task
    assert status["worker"]["source"] == "worker"
    assert status["web"]["source"] == "web"


def test_modal_subprocess_environment_forces_utf8_without_mutating_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PYTHONUTF8", raising=False)
    monkeypatch.delenv("PYTHONIOENCODING", raising=False)

    child_env = ModalService._subprocess_env("utf8-profile")

    assert child_env["MODAL_PROFILE"] == "utf8-profile"
    assert child_env["PYTHONUTF8"] == "1"
    assert child_env["PYTHONIOENCODING"] == "utf-8"
    assert "PYTHONUTF8" not in os.environ
    assert "PYTHONIOENCODING" not in os.environ


def test_modal_install_progress_events_update_public_snapshot(tmp_path: Path) -> None:
    service = ModalService(tmp_path, lambda: {})
    started_at = 100.0
    service._install_state = {
        "state": "running",
        "phase": "upload",
        "started_at": started_at,
        "progress": {
            "mode": "determinate",
            "completed_files": 0,
            "total_files": 3,
            "completed_bytes": 0,
            "total_bytes": 30,
        },
        "logs": [],
    }

    service._handle_install_client_output(
        "stderr",
        INSTALL_PROGRESS_PREFIX
        + json.dumps(
            {
                "event": "batch_complete",
                "label": "모델",
                "processed_files": 2,
                "processed_bytes": 20,
                "uploaded_files": 1,
                "skipped_files": 1,
            },
            ensure_ascii=False,
        ),
    )
    snapshot = service._install_snapshot()

    assert snapshot["progress"]["completed_files"] == 2
    assert snapshot["progress"]["completed_bytes"] == 20
    assert snapshot["progress"]["uploaded_files"] == 1
    assert snapshot["progress"]["skipped_files"] == 1
    assert snapshot["logs"][-1]["source"] == "upload"
    assert "모델 완료" in snapshot["logs"][-1]["message"]


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
    commands: list[list[str]] = []

    async def connected(_settings: ModalSettings) -> bool:
        return True

    async def capture_commands(_args, **kwargs):
        commands.append(list(_args))
        payload = kwargs.get("stdin_payload")
        if isinstance(payload, dict):
            observed.update(payload)
            return 0, json.dumps({"ok": True, "result": {}}), ""
        return 0, "", ""

    monkeypatch.setattr(service, "account_connected", connected)
    monkeypatch.setattr(service, "_run_command", capture_commands)

    # 파일명(확장자 포함)으로 동기화 대상을 지정한다. config 바인딩 아님.
    await service.start_install([workflow.name])
    assert service._install_task is not None
    await service._install_task

    assert service._install_state["state"] == "completed"
    assert observed["action"] == "install"
    assert observed["model_files"][0]["source_path"] == str(checkpoint.resolve())
    assert observed["model_files"][0]["remote_path"] == "checkpoints/changed.safetensors"
    assert observed["lora_files"] == []
    deploy_commands = [
        command
        for command in commands
        if len(command) > 4 and command[1:4] == ["-m", "modal", "deploy"]
    ]
    assert deploy_commands == []
    assert commands == [[sys.executable, "-m", "modal_backend.client_cli"]]


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


def test_workflow_assets_resolve_soya_prompt_parser_cache_inputs(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    character_root = input_root / "soya_bot" / "sample_bot" / "alice"
    character_root.mkdir(parents=True)
    cache_pt = character_root / "cache.pt"
    cache_ipadapter = character_root / "cache.ipadpt"
    cache_pt.write_bytes(b"embedding-cache")
    cache_ipadapter.write_bytes(b"ipadapter-cache")
    positive = "\n".join(
        [
            "[CHAR_LIST]",
            "alice",
            "[CACHE_PATH]",
            json.dumps(
                {"list": [{"emb_path": "soya_bot/sample_bot/alice/cache.pt"}]}
            ),
            "[FACE_ID_DIR]",
            json.dumps(
                {"list": [{"ipa_path": "soya_bot/sample_bot/alice/cache.ipadpt"}]}
            ),
        ]
    )
    workflow = {
        "9": {"class_type": "PrimitiveStringMultiline", "inputs": {"value": positive}},
        "909": {"class_type": "SoyaPromptParser_mdsoya", "inputs": {"text": ["9", 0]}},
        "458": {
            "class_type": "SoyaIPAPatchMaker_mdsoya",
            "inputs": {
                "embed_cache_data": ["909", 8],
                "ipa_cache_data": ["909", 10],
            },
        },
    }

    inputs = resolve_input_files(workflow, {"comfy_input_dir": str(input_root)})

    assert inputs == [
        {
            "source_path": str(cache_pt),
            "remote_name": "soya_bot/sample_bot/alice/cache.pt",
        },
        {
            "source_path": str(cache_ipadapter),
            "remote_name": "soya_bot/sample_bot/alice/cache.ipadpt",
        },
    ]


def test_workflow_assets_resolve_direct_soya_cache_inputs(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    cache_file = input_root / "soya_bot" / "sample_bot" / "alice" / "cache.pt"
    cache_file.parent.mkdir(parents=True)
    cache_file.write_bytes(b"embedding-cache")
    workflow = {
        "458": {
            "class_type": "SoyaIPAPatchMaker_mdsoya",
            "inputs": {
                "embed_cache_data": json.dumps(
                    {"list": [{"emb_path": "soya_bot/sample_bot/alice/cache.pt"}]}
                )
            },
        }
    }

    inputs = resolve_input_files(workflow, {"comfy_input_dir": str(input_root)})

    assert inputs == [
        {
            "source_path": str(cache_file),
            "remote_name": "soya_bot/sample_bot/alice/cache.pt",
        }
    ]


def test_workflow_assets_reject_missing_soya_cache_before_remote_call(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "input"
    input_root.mkdir()
    workflow = {
        "458": {
            "class_type": "SoyaIPAPatchMaker_mdsoya",
            "inputs": {
                "embed_cache_data": json.dumps(
                    {"list": [{"emb_path": "soya_bot/sample_bot/alice/cache.pt"}]}
                )
            },
        }
    }

    with pytest.raises(FileNotFoundError, match="필수 캐시 파일이 없습니다"):
        resolve_input_files(workflow, {"comfy_input_dir": str(input_root)})


def test_workflow_assets_reject_invalid_soya_cache_json(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    input_root.mkdir()
    workflow = {
        "458": {
            "class_type": "SoyaIPAPatchMaker_mdsoya",
            "inputs": {"embed_cache_data": '{"list": ['},
        }
    }

    with pytest.raises(ValueError, match="캐시 JSON 형식이 올바르지 않습니다"):
        resolve_input_files(workflow, {"comfy_input_dir": str(input_root)})


def test_workflow_assets_reject_soya_cache_outside_input_root(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    input_root.mkdir()
    workflow = {
        "458": {
            "class_type": "SoyaIPAPatchMaker_mdsoya",
            "inputs": {
                "embed_cache_data": json.dumps(
                    {"list": [{"emb_path": "../outside/cache.pt"}]}
                )
            },
        }
    }

    with pytest.raises(ValueError, match="안전하지 않은 필수 캐시 입력 상대 경로"):
        resolve_input_files(workflow, {"comfy_input_dir": str(input_root)})


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
async def test_modal_worker_status_skips_remote_lookup_when_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = ModalService(tmp_path, lambda: {"modal_enabled": False})

    async def unexpected_client_action(*_args, **_kwargs) -> dict:
        raise AssertionError("Modal OFF 상태에서는 runtime_stats를 조회하면 안 됩니다.")

    monkeypatch.setattr(service, "_run_client_action", unexpected_client_action)

    status = await service.worker_status()
    worker = status["worker"]

    assert status["ok"] is True
    assert status["enabled"] is False
    assert worker["available"] is False
    assert worker["reason"] == "disabled"
    assert worker["gpu_on"] is False
    assert worker["workers"] == 0


@pytest.mark.asyncio
async def test_modal_worker_status_returns_lightweight_runtime_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = ModalService(
        tmp_path,
        lambda: {
            "modal_enabled": True,
            "modal_worker_gpu": "L40S",
            "modal_web_gpu": "RTX-PRO-6000",
            "modal_status_refresh_seconds": 7,
        },
    )
    observed: dict = {}

    async def client_action(
        settings: ModalSettings,
        action: str,
        *,
        timeout: float,
        **_payload,
    ) -> dict:
        observed[action] = {
            "worker_gpu": settings.worker_gpu,
            "web_gpu": settings.web_gpu,
            "timeout": timeout,
        }
        if action == "runtime_stats":
            return {
                "num_total_runners": 2,
                "num_running_inputs": 1,
                "backlog": 3,
                "input_headroom": 0,
            }
        if action == "web_status":
            return {
                "url": "https://worker-app-web.modal.run",
                "num_total_runners": 0,
            }
        raise AssertionError(f"예상하지 못한 Modal client action: {action}")

    monkeypatch.setattr(service, "_run_client_action", client_action)

    status = await service.worker_status()

    assert observed["runtime_stats"] == {
        "worker_gpu": "L40S",
        "web_gpu": "RTX-PRO-6000",
        "timeout": 30,
    }
    assert observed["web_status"] == {
        "worker_gpu": "L40S",
        "web_gpu": "RTX-PRO-6000",
        "timeout": 30,
    }
    assert status["ok"] is True
    assert status["enabled"] is True
    assert status["gpu"] == "L40S"
    assert status["worker_gpu"] == "L40S"
    assert status["web_gpu"] == "RTX-PRO-6000"
    assert status["refresh_seconds"] == 7
    assert status["worker"]["available"] is True
    assert status["worker"]["gpu_on"] is True
    assert status["worker"]["workers"] == 2
    assert status["worker"]["generating"] == 1
    assert status["worker"]["queued"] == 3
    assert status["web"]["gpu"] == "RTX-PRO-6000"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reason", "error_type"),
    [
        ("app_not_deployed", "NotFoundError"),
        ("network_unavailable", "ConnectionError"),
    ],
)
async def test_modal_worker_status_exposes_specific_unavailable_reason(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reason: str,
    error_type: str,
) -> None:
    service = ModalService(tmp_path, lambda: {"modal_enabled": True})

    async def unavailable_client_action(*_args, **_kwargs) -> dict:
        raise ModalClientActionError(
            "Modal runtime unavailable",
            reason=reason,
            error_type=error_type,
        )

    monkeypatch.setattr(service, "_run_client_action", unavailable_client_action)

    status = await service.worker_status()
    worker = status["worker"]

    assert status["ok"] is True
    assert status["enabled"] is True
    assert worker["available"] is False
    assert worker["reason"] == reason
    assert worker["gpu_on"] is False
    assert worker["workers"] == 0


@pytest.mark.asyncio
async def test_modal_worker_status_does_not_treat_volume_sync_as_deployment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = ModalService(tmp_path, lambda: {"modal_enabled": True})
    service._install_state = {
        "state": "running",
        "phase": "assets",
        "message": "동기화할 자산을 분석하고 있습니다.",
        "logs": [],
    }

    async def unavailable_client_action(*_args, **_kwargs) -> dict:
        raise ModalClientActionError(
            "Modal 작업 App이 배포되지 않았습니다.",
            reason="app_not_deployed",
            error_type="NotFoundError",
        )

    monkeypatch.setattr(service, "_run_client_action", unavailable_client_action)

    status = await service.worker_status()
    worker = status["worker"]

    assert worker["available"] is False
    assert worker["state"] == "error"
    assert worker["reason"] == "app_not_deployed"
    assert worker["install_phase"] is None
    assert "배포되지 않았습니다" in worker["error"]


@pytest.mark.asyncio
async def test_modal_web_url_returns_public_url_when_deployed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = ModalService(tmp_path, lambda: {"modal_enabled": True, "modal_gpu": "L4"})
    observed: dict = {}

    async def connected(_settings: ModalSettings) -> bool:
        return True

    async def client_action(
        settings: ModalSettings,
        action: str,
        *,
        timeout: float,
        **_payload,
    ) -> dict:
        observed.update({"action": action, "timeout": timeout, **_payload})
        return {
            "url": "https://workspace--soya-comfy-worker-web-comfy-web-server.modal.run",
            "num_total_runners": 1,
            "num_running_inputs": 0,
            "backlog": 0,
        }

    monkeypatch.setattr(service, "account_connected", connected)
    monkeypatch.setattr(service, "_run_client_action", client_action)

    result = await service.web_url()

    assert observed == {
        "action": "web_status",
        "timeout": 30,
        "web_app_name": "soya-comfy-worker-web",
    }
    assert result == {
        "available": True,
        "state": "running",
        "url": "https://workspace--soya-comfy-worker-web-comfy-web-server.modal.run",
        "app_name": "soya-comfy-worker-web",
    }


@pytest.mark.asyncio
async def test_modal_web_url_reports_app_not_deployed_when_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = ModalService(tmp_path, lambda: {"modal_enabled": True, "modal_gpu": "L4"})

    async def connected(_settings: ModalSettings) -> bool:
        return True

    async def not_deployed(*_args, **_kwargs) -> dict:
        raise ModalClientActionError(
            "comfy_web_server not found",
            reason="app_not_deployed",
            error_type="NotFoundError",
        )

    monkeypatch.setattr(service, "account_connected", connected)
    monkeypatch.setattr(service, "_run_client_action", not_deployed)

    result = await service.web_url()

    assert result["available"] is False
    assert result["reason"] == "app_not_deployed"
    assert "comfy_web_server not found" in result["error"]


@pytest.mark.asyncio
async def test_modal_web_url_treats_none_url_as_not_deployed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = ModalService(tmp_path, lambda: {"modal_enabled": True, "modal_gpu": "L4"})

    async def connected(_settings: ModalSettings) -> bool:
        return True

    async def returns_none(*_args, **_kwargs) -> dict:
        return {"url": None}

    monkeypatch.setattr(service, "account_connected", connected)
    monkeypatch.setattr(service, "_run_client_action", returns_none)

    result = await service.web_url()

    assert result["available"] is False
    assert result["reason"] == "app_not_deployed"
    assert result["state"] == "stopped"


@pytest.mark.asyncio
async def test_modal_web_url_skips_remote_lookup_when_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = ModalService(tmp_path, lambda: {"modal_enabled": False})

    async def unexpected_client_action(*_args, **_kwargs) -> dict:
        raise AssertionError("Modal OFF 상태에서는 웹 URL 원격 조회를 하면 안 됩니다.")

    monkeypatch.setattr(service, "_run_client_action", unexpected_client_action)

    result = await service.web_url()

    assert result["available"] is False
    assert result["reason"] == "disabled"
    assert result["state"] == "stopped"


@pytest.mark.asyncio
async def test_modal_web_url_requires_account_connection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = ModalService(tmp_path, lambda: {"modal_enabled": True, "modal_gpu": "L4"})

    async def disconnected(_settings: ModalSettings) -> bool:
        return False

    async def unexpected_client_action(*_args, **_kwargs) -> dict:
        raise AssertionError("계정 미연결 상태에서는 웹 URL 원격 조회를 하면 안 됩니다.")

    monkeypatch.setattr(service, "account_connected", disconnected)
    monkeypatch.setattr(service, "_run_client_action", unexpected_client_action)

    result = await service.web_url()

    assert result["available"] is False
    assert result["reason"] == "account_not_connected"
    assert result["state"] == "stopped"


@pytest.mark.asyncio
async def test_modal_client_web_url_action_reads_comfy_web_server_url(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "action": "web_url",
        "app_name": "soya-comfy-worker",
        "environment": "main",
    }
    captured: dict = {}

    class FakeServer:
        def get_url(self) -> str | None:
            return "https://workspace--soya-comfy-worker-comfy-web-server.modal.run"

    def from_name(app_name: str, server_name: str, *, environment_name: str):
        captured.update(
            {"app_name": app_name, "server_name": server_name, "env": environment_name}
        )
        assert server_name == "comfy_web_server"
        return FakeServer()

    monkeypatch.setattr(client_cli.modal.Server, "from_name", from_name)

    result = client_cli.web_url(payload)

    assert captured == {
        "app_name": "soya-comfy-worker-web",
        "server_name": "comfy_web_server",
        "env": "main",
    }
    assert result == {
        "url": "https://workspace--soya-comfy-worker-comfy-web-server.modal.run"
    }


def test_modal_client_web_status_reads_dedicated_web_app_stats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "app_name": "worker",
        "web_app_name": "worker-manual-web",
        "environment": "main",
    }
    captured: dict = {}

    def server_status(actual_payload: dict) -> dict:
        captured.update(actual_payload)
        return {
            "url": "https://example.modal.run",
            "deployed": True,
            "runners": 1,
            "app_name": "worker-manual-web",
        }

    monkeypatch.setattr(client_cli, "_web_server_status", server_status)

    result = client_cli.web_status(payload)

    assert captured == {
        "app_name": "worker",
        "web_app_name": "worker-manual-web",
        "environment": "main",
    }
    assert result["num_total_runners"] == 1
    assert result["num_running_inputs"] == 0
    assert result["backlog"] == 0
    assert result["deployed"] is True
    assert result["runners"] == 1


def test_modal_client_web_server_status_reads_url_and_deployed_app_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict = {}

    class FakeStub:
        async def AppList(self, request):
            observed["environment"] = request.environment_name
            return SimpleNamespace(
                apps=[
                    SimpleNamespace(
                        description="worker-web",
                        state=client_cli.api_pb2.APP_STATE_DEPLOYED,
                        n_running_tasks=2,
                    ),
                    SimpleNamespace(
                        description="worker-web",
                        state=client_cli.api_pb2.APP_STATE_STOPPED,
                        n_running_tasks=0,
                    ),
                ]
            )

    class FakeClient:
        stub = FakeStub()

    class FakeServer:
        async def get_url(self) -> str:
            return "https://example.modal.direct"

    async def from_env():
        return FakeClient()

    def from_name(
        app_name: str,
        server_name: str,
        *,
        environment_name: str,
        client,
    ) -> FakeServer:
        observed["app_name"] = app_name
        observed["server_name"] = server_name
        observed["server_environment"] = environment_name
        assert isinstance(client, FakeClient)
        return FakeServer()

    monkeypatch.setattr(client_cli._Client, "from_env", from_env)
    monkeypatch.setattr(client_cli._Server, "from_name", from_name)

    result = client_cli._web_server_status(
        {
            "app_name": "worker",
            "web_app_name": "worker-web",
            "environment": "main",
        }
    )

    assert observed == {
        "environment": "main",
        "app_name": "worker-web",
        "server_name": "comfy_web_server",
        "server_environment": "main",
    }
    assert result == {
        "url": "https://example.modal.direct",
        "deployed": True,
        "runners": 2,
        "app_name": "worker-web",
    }


def test_modal_client_web_status_treats_missing_app_as_stopped(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = {
        "app_name": "worker",
        "web_app_name": "worker-web",
        "environment": "main",
    }

    class FakeClient:
        pass

    class MissingServer:
        async def get_url(self) -> str:
            raise client_cli.modal.exception.NotFoundError("missing web app")

    async def from_env():
        return FakeClient()

    def from_name(
        app_name: str,
        server_name: str,
        *,
        environment_name: str,
        client,
    ) -> MissingServer:
        assert app_name == "worker-web"
        assert server_name == "comfy_web_server"
        assert environment_name == "main"
        assert isinstance(client, FakeClient)
        return MissingServer()

    monkeypatch.setattr(client_cli._Client, "from_env", from_env)
    monkeypatch.setattr(client_cli._Server, "from_name", from_name)

    result = client_cli.web_status(payload)

    assert result == {
        "url": None,
        "deployed": False,
        "runners": None,
        "app_name": "worker-web",
        "backlog": 0,
        "num_total_runners": 0,
        "num_running_inputs": 0,
    }
    captured = capsys.readouterr()
    assert "missing web app" in captured.err
    assert "Traceback" not in captured.err


@pytest.mark.asyncio
async def test_modal_web_stop_only_stops_dedicated_web_app(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = ModalService(
        tmp_path,
        lambda: {
            "modal_enabled": True,
            "modal_deployment_name": "worker-app",
            "modal_environment": "main",
        },
    )
    observed: dict = {}

    async def logs(*, entries: int) -> dict:
        observed["log_entries"] = entries
        return {"ok": True, "logs": [], "errors": []}

    async def run_command(args, **_kwargs):
        observed["command"] = list(args)
        return 0, "", ""

    monkeypatch.setattr(service, "runtime_logs", logs)
    monkeypatch.setattr(service, "_run_command", run_command)

    settings = ModalSettings.from_mapping(service.get_config())
    await service._run_web_stop(settings)

    assert observed["log_entries"] == 500
    assert observed["command"][1:4] == ["-m", "modal", "app"]
    assert observed["command"][4:7] == ["stop", "worker-app-web", "--yes"]
    assert service._web_state["state"] == "stopped"
    assert service._web_state["app_name"] == "worker-app-web"


@pytest.mark.asyncio
async def test_modal_web_stop_keeps_stopped_when_log_collection_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    service = ModalService(
        tmp_path,
        lambda: {
            "modal_enabled": True,
            "modal_deployment_name": "worker-app",
        },
    )
    observed: list[str] = []

    async def stop(_settings: ModalSettings) -> None:
        observed.append("stop")

    async def logs(*, entries: int) -> dict:
        observed.append(f"logs:{entries}")
        raise RuntimeError("log service unavailable")

    monkeypatch.setattr(service, "_stop_web_app", stop)
    monkeypatch.setattr(service, "runtime_logs", logs)

    settings = ModalSettings.from_mapping(service.get_config())
    await service._run_web_stop(settings)

    assert observed == ["stop", "logs:500"]
    assert service._web_state["state"] == "stopped"
    assert service._web_state["deployed"] is False
    captured = capsys.readouterr()
    assert "종료 후 로그 조회 실패" in captured.out
    assert "Traceback" in captured.err


@pytest.mark.asyncio
async def test_modal_streaming_command_cancellation_kills_child_process() -> None:
    ready = asyncio.Event()

    def output_callback(_source: str, line: str) -> None:
        if line == "ready":
            ready.set()

    task = asyncio.create_task(
        ModalService._run_command(
            [
                sys.executable,
                "-u",
                "-c",
                "import time; print('ready', flush=True); time.sleep(300)",
            ],
            env=dict(os.environ),
            timeout=310,
            output_callback=output_callback,
        )
    )
    await asyncio.wait_for(ready.wait(), timeout=10)

    started = time.monotonic()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=5)

    assert time.monotonic() - started < 5


@pytest.mark.asyncio
async def test_modal_web_deploy_uses_package_module_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = ModalService(
        tmp_path,
        lambda: {
            "modal_enabled": True,
            "modal_deployment_name": "worker-app",
            "modal_environment": "main",
            "modal_worker_gpu": "RTX-PRO-6000",
            "modal_web_gpu": "L40S",
            "modal_web_fast": True,
        },
    )
    observed: dict = {}

    async def run_command(args, **kwargs):
        observed["args"] = list(args)
        observed["kwargs"] = kwargs
        return 0, "", ""

    monkeypatch.setattr(service, "_run_command", run_command)

    settings = ModalSettings.from_mapping(service.get_config())
    await service._deploy_web_app(settings, custom_node_inventory={})

    assert observed["args"][1:7] == [
        "-m",
        "modal",
        "deploy",
        "-m",
        "modal_backend.modal_web_app",
        "--env",
    ]
    assert observed["args"][7] == "main"
    assert observed["kwargs"]["timeout"] == 3600
    assert observed["kwargs"]["env"]["SOYA_MODAL_WEB_FAST"] == "1"
    assert observed["kwargs"]["env"]["SOYA_MODAL_WORKER_GPU"] == "RTX-PRO-6000"
    assert observed["kwargs"]["env"]["SOYA_MODAL_WEB_GPU"] == "L40S"


@pytest.mark.asyncio
async def test_modal_web_start_redeploys_web_app_before_warming_l4(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = ModalService(
        tmp_path,
        lambda: {"modal_enabled": True, "modal_deployment_name": "worker-app"},
    )
    settings = ModalSettings.from_mapping(service.get_config())
    observed: list[str] = []

    async def deploy(_settings: ModalSettings) -> None:
        observed.append("deploy")

    async def status(_settings: ModalSettings) -> dict:
        observed.append("status")
        return {
            "available": True,
            "deployed": True,
            "state": "running",
            "url": "https://example.modal.run",
            "app_name": "worker-app-web",
            "num_total_runners": 1,
            "num_running_inputs": 0,
            "backlog": 0,
        }

    def warm(url: str, _cancel_event: threading.Event | None = None) -> int:
        observed.append(f"warm:{url}")
        return 200

    monkeypatch.setattr(service, "_deploy_web_app", deploy)
    monkeypatch.setattr(service, "_remote_web_status", status)
    monkeypatch.setattr(service, "_warm_web_url", warm)

    await service._run_web_start(settings, {"deployed": True})

    assert observed == [
        "deploy",
        "status",
        "warm:https://example.modal.run",
        "status",
    ]
    assert service._web_state["state"] == "running"


def test_modal_web_warm_retries_server_cold_start_503(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[float] = []
    sleeps: list[float] = []

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def read(self, _size: int) -> bytes:
            return b"<"

    def urlopen(request, *, timeout: float):
        calls.append(timeout)
        if len(calls) < 3:
            raise urllib.error.HTTPError(
                request.full_url,
                503,
                "cold start",
                hdrs=None,
                fp=None,
            )
        return FakeResponse()

    monkeypatch.setattr("modal_backend.service.urllib.request.urlopen", urlopen)
    monkeypatch.setattr(
        "modal_backend.service.time.sleep",
        lambda seconds: sleeps.append(seconds),
    )

    result = ModalService._warm_web_url("https://example.modal.run")

    assert result == 200
    assert len(calls) == 3
    assert sleeps == [1, 1]


def test_modal_web_warm_stops_before_another_request_when_cancelled(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cancel_event = threading.Event()
    cancel_event.set()

    def unexpected_urlopen(*_args, **_kwargs):
        raise AssertionError("취소 후에는 웹 Server를 다시 호출하면 안 됩니다.")

    monkeypatch.setattr(
        "modal_backend.service.urllib.request.urlopen",
        unexpected_urlopen,
    )

    with pytest.raises(WebStartCancelled):
        ModalService._warm_web_url(
            "https://example.modal.run",
            cancel_event,
        )

    assert "웹 Server 준비 대기 취소" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_modal_web_start_failure_stops_web_app(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = ModalService(
        tmp_path,
        lambda: {
            "modal_enabled": True,
            "modal_deployment_name": "worker-app",
            "modal_environment": "main",
        },
    )
    settings = ModalSettings.from_mapping(service.get_config())
    observed: list[str] = []

    async def deploy(_settings: ModalSettings) -> None:
        observed.append("deploy")
        raise RuntimeError("broken deployment")

    async def stop(_settings: ModalSettings) -> None:
        observed.append("stop")

    monkeypatch.setattr(service, "_deploy_web_app", deploy)
    monkeypatch.setattr(service, "_stop_web_app", stop)

    await service._run_web_start(settings, {"deployed": False})

    assert observed == ["deploy", "stop"]
    assert service._web_state["state"] == "failed"
    assert service._web_state["deployed"] is False
    assert service._web_state["num_total_runners"] == 0
    assert "자동 종료" in service._web_state["message"]


@pytest.mark.asyncio
async def test_modal_web_stop_cancels_starting_task_before_stopping_app(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = ModalService(
        tmp_path,
        lambda: {
            "modal_enabled": True,
            "modal_deployment_name": "worker-app",
            "modal_environment": "main",
        },
    )
    observed: list[str] = []
    wait_forever = asyncio.Event()

    async def pending_start() -> None:
        try:
            await wait_forever.wait()
        finally:
            observed.append("start_cancelled")

    async def logs(*, entries: int) -> dict:
        observed.append(f"logs:{entries}")
        return {"ok": True, "logs": [], "errors": []}

    async def stop(_settings: ModalSettings) -> None:
        observed.append("stop")

    async def unexpected_connection_check(_settings: ModalSettings) -> bool:
        raise AssertionError("시작 취소는 계정 재확인을 기다리면 안 됩니다.")

    monkeypatch.setattr(service, "runtime_logs", logs)
    monkeypatch.setattr(service, "_stop_web_app", stop)
    monkeypatch.setattr(service, "account_connected", unexpected_connection_check)

    start_task = asyncio.create_task(pending_start())
    await asyncio.sleep(0)
    service._web_task = start_task
    service._web_state = {
        "available": True,
        "deployed": True,
        "state": "starting",
        "message": "L4 준비 중",
    }

    result = await service.stop_web()
    stop_task = service._web_task
    assert stop_task is not None
    await stop_task

    assert result["state"] == "stopping"
    assert "시작을 취소" in result["message"]
    assert service._web_start_cancel_event.is_set()
    assert start_task.cancelled()
    assert observed == ["start_cancelled", "stop", "logs:500"]
    assert service._web_state["state"] == "stopped"
    assert service._web_state["deployed"] is False


@pytest.mark.asyncio
async def test_modal_runtime_logs_merge_remote_sync_and_diagnostic_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = ModalService(tmp_path, lambda: {"modal_enabled": True})
    service._runtime_log_session_started_at = 5.0
    service._install_state["logs"] = [
        {"time": 20.0, "source": "upload", "message": "워크플로우 업로드 완료"}
    ]
    service._probe_state = {
        "state": "completed",
        "message": "L4 · VRAM 24 GiB 연결 확인",
        "updated_at": 30.0,
    }

    async def connected(_settings: ModalSettings) -> bool:
        return True

    async def client_action(*_args, **_kwargs) -> dict:
        return {
            "logs": [
                {
                    "time": 10.0,
                    "source": "stdout",
                    "category": "jobs",
                    "app_name": "soya-comfy-worker",
                    "message": "prompt complete",
                }
            ],
            "errors": [],
        }

    monkeypatch.setattr(service, "account_connected", connected)
    monkeypatch.setattr(service, "_run_client_action", client_action)

    result = await service.runtime_logs(entries=100)

    assert [entry["category"] for entry in result["logs"]] == [
        "jobs",
        "sync",
        "diagnostic",
    ]
    assert result["errors"] == []


@pytest.mark.asyncio
async def test_modal_runtime_logs_only_include_current_server_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = ModalService(tmp_path, lambda: {"modal_enabled": True})
    service._runtime_log_session_started_at = 100.0
    service._install_state["logs"] = [
        {"time": 90.0, "source": "upload", "message": "이전 동기화"},
        {"time": 102.0, "source": "upload", "message": "현재 동기화"},
    ]
    service._probe_state = {
        "state": "idle",
        "message": "작업 워커 GPU 연결 테스트를 기다리고 있습니다.",
        "updated_at": 103.0,
    }

    async def connected(_settings: ModalSettings) -> bool:
        return True

    async def client_action(*_args, **_kwargs) -> dict:
        return {
            "logs": [
                {
                    "time": 99.0,
                    "source": "stdout",
                    "category": "web",
                    "app_name": "soya-comfy-worker",
                    "message": "이전 웹 로그",
                },
                {
                    "time": 101.0,
                    "source": "stdout",
                    "category": "web",
                    "app_name": "soya-comfy-worker-web",
                    "message": "현재 웹 로그",
                },
            ],
            "errors": [],
        }

    monkeypatch.setattr(service, "account_connected", connected)
    monkeypatch.setattr(service, "_run_client_action", client_action)

    result = await service.runtime_logs(entries=100)

    assert [entry["message"] for entry in result["logs"]] == [
        "현재 웹 로그",
        "현재 동기화",
    ]
    assert all(
        entry["message"] != "작업 워커 GPU 연결 테스트를 기다리고 있습니다."
        for entry in result["logs"]
    )


def test_modal_web_server_is_isolated_from_worker_app() -> None:
    root = Path(__file__).resolve().parents[1] / "modal_backend"
    worker_source = (root / "modal_app.py").read_text(encoding="utf-8")
    web_source = (root / "modal_web_app.py").read_text(encoding="utf-8")
    frontend_source = (
        Path(__file__).resolve().parents[1] / "frontend" / "index.html"
    ).read_text(encoding="utf-8")

    assert "class ComfyWebServer" not in worker_source
    assert "class ComfyWebServer" in web_source
    assert "@app.server(" in web_source
    assert "@modal.web_server" not in web_source
    assert 'name="comfy_web_server"' in web_source
    assert "unauthenticated=True" in web_source
    assert "@modal.enter()" in web_source
    assert 'WEB_APP_NAME = os.environ.get("SOYA_MODAL_WEB_APP_NAME"' in web_source
    assert 'gpu=WORKER_GPU' not in worker_source
    assert "worker_cls.with_options(gpu=_worker_gpu(payload))" in (
        root / "client_cli.py"
    ).read_text(encoding="utf-8")
    assert 'gpu=WEB_GPU' in web_source
    assert (
        'WEB_WORKFLOW_MOUNT_PATH = '
        '"/root/ComfyUI/user/default/workflows/SOYA_USER"'
    ) in web_source
    assert 'WEB_WORKFLOW_MOUNT_PATH: workflows_volume' in web_source
    assert '"/workflows": workflows_volume' in worker_source
    assert '"/workflows": workflows_volume' not in web_source
    assert 'COMFY_MODELS_MOUNT_PATH = "/root/ComfyUI/models"' in worker_source
    assert 'COMFY_MODELS_MOUNT_PATH: models_volume' in worker_source
    assert 'COMFY_MODELS_MOUNT_PATH: models_volume' in web_source
    assert '"/models": models_volume' not in worker_source
    assert '"/models": models_volume' not in web_source
    assert 'f"  base_path: {COMFY_MODELS_MOUNT_PATH}"' in worker_source
    cleanup_command = 'f"rm -rf {COMFY_MODELS_MOUNT_PATH}"'
    assert cleanup_command in worker_source
    assert worker_source.index('"python /opt/soya/image_install.py"') < worker_source.index(
        cleanup_command
    )
    assert '"--enable-cors-header",\n            "*",' in web_source
    assert 'WEB_FAST = os.environ.get("SOYA_MODAL_WEB_FAST", "0") == "1"' in web_source
    assert 'web_runtime_image = runtime_image.env(' in web_source
    assert 'if web_fast:\n            command.append("--fast")' in web_source
    assert (
        "const showWebStopAction = webState === 'starting' || webRunning || "
        "webState === 'stopping';"
    ) in frontend_source
    assert "? 'ComfyUI 시작 취소'" in frontend_source


def test_modal_runtime_image_installs_prebuilt_sageattention_wheel() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "modal_backend" / "modal_app.py"
    ).read_text(encoding="utf-8")

    assert 'CUDA_VERSION = "12.8.1"' in source
    assert 'PYTHON_VERSION = "3.12"' in source
    assert 'TORCH_VERSION = "2.11.0"' in source
    assert 'TORCHVISION_VERSION = "0.26.0"' in source
    assert 'TORCHAUDIO_VERSION = "2.11.0"' in source
    assert 'SAGEATTENTION_VERSION = "2.2.0"' in source
    assert "modal.Image.from_registry(" in source
    assert 'f"nvidia/cuda:{CUDA_VERSION}-devel-ubuntu22.04"' in source
    assert "add_python=PYTHON_VERSION" in source
    assert ".entrypoint([])" in source
    assert '"CUDA_HOME": "/usr/local/cuda"' in source
    assert '"CC": "/usr/bin/gcc"' in source
    assert '"CXX": "/usr/bin/g++"' in source
    assert '"CUDAHOSTCXX": "/usr/bin/g++"' in source
    assert 'SAGEATTENTION_WHEEL_URL = (' in source
    assert '"sageattention-2.2.0%2Bcu128torch2.11-cp312-cp312-"' in source
    assert '"manylinux_2_34_x86_64.manylinux_2_35_x86_64.whl"' in source
    assert (
        'SAGEATTENTION_WHEEL_SHA256 = (\n'
        '    "900c20a9baa591463731da9a25f626587ebb1902d2c902a494bfacb9fe8981fc"'
    ) in source
    assert "#sha256={SAGEATTENTION_WHEEL_SHA256}" in source
    assert 'gpu=WORKER_GPU' not in source
    assert {
        str(profile["cuda_arch"])
        for profile in ModalSettings.from_mapping({}).public_dict()["gpu_profiles"]
    } == {"8.9", "12.0"}
    assert '"TORCH_CUDA_ARCH_LIST"' not in source
    assert '"MAX_JOBS"' not in source
    assert '"EXT_PARALLEL"' not in source
    assert "index_url=PYTORCH_CUDA_INDEX_URL" in source
    assert "https://github.com/thu-ml/SageAttention.git" not in source
    assert "sageattention_build.py" not in source
    assert "--no-build-isolation" not in source
    assert "torch.version.cuda == '12.8'" in source
    assert "pv.Version(m.version('sageattention')).base_version" in source
    assert "modal.Image.debian_slim" not in source
    assert source.index('f"torch=={TORCH_VERSION}"') < source.index(
        "#sha256={SAGEATTENTION_WHEEL_SHA256}"
    )
    assert source.index(
        "#sha256={SAGEATTENTION_WHEEL_SHA256}"
    ) < source.index('"python /opt/soya/image_install.py"')
    assert "def _validate_sageattention_cuda()" in source
    assert "output = sageattn(" in source
    assert "torch.cuda.synchronize()" in source
    assert "torch.isfinite(output).all().item()" in source


def test_modal_worker_uses_gpu_memory_snapshot_without_warm_pool() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "modal_backend" / "modal_app.py"
    ).read_text(encoding="utf-8")
    worker_definition = source[source.index("@app.cls(") : source.index("    @modal.method()")]

    assert "min_containers=0" in worker_definition
    assert "enable_memory_snapshot=True" in worker_definition
    assert 'experimental_options={"enable_gpu_snapshot": True}' in worker_definition
    assert 'env={"SOYA_MODAL_SNAPSHOT_VERSION": SNAPSHOT_VERSION}' in worker_definition
    assert "@modal.enter(snap=True)\n    def start" in worker_definition
    assert "@modal.enter(snap=False)\n    def restore" in worker_definition
    assert "self.text_output_thread.is_alive()" in worker_definition
    assert "GPU Memory Snapshot 복원 완료" in worker_definition


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
    remote_workflow = {
        "nodes": [{"id": 7, "type": "RemoteOnlyNode"}],
        "links": [],
    }
    actions: list[str] = []

    async def client_action(
        _settings: ModalSettings,
        action: str,
        *,
        timeout: float,
        **payload,
    ) -> dict:
        actions.append(action)
        if action == "read_workflow":
            assert timeout == 120
            assert payload["workflow_name"] == workflow.name
            return {
                "name": workflow.name,
                "sha256": "a" * 64,
                "workflow": remote_workflow,
            }
        assert action == "convert_workflow"
        assert timeout == 960
        assert payload["workflow"] == remote_workflow
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

    # 파일명(확장자 포함)으로 실행 대상을 지정한다. config 바인딩 아님.
    state = await service.start_workflow_run(workflow.name)
    await service._workflow_run_tasks[state["job_id"]]

    completed = service.workflow_run_status(state["job_id"])
    image, content_type = service.workflow_run_image(state["job_id"])
    assert completed["state"] == "completed"
    assert completed["remote_sha256"] == "a" * 64
    assert completed["result_available"] is True
    assert completed["model_sync"] == {"uploaded": 0}
    assert "image_bytes" not in completed
    assert image == b"png"
    assert content_type == "image/png"
    assert actions == ["read_workflow", "convert_workflow"]


@pytest.mark.asyncio
async def test_managed_workflow_run_rejects_missing_remote_workflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root, user_root = _modal_test_project(tmp_path)
    workflow = user_root / "local-only.json"
    workflow.write_text('{"1":{"class_type":"EmptyLatentImage"}}', encoding="utf-8")
    service = ModalService(project_root, lambda: {"modal_enabled": True})

    async def connected(_settings: ModalSettings) -> bool:
        return True

    async def missing_remote(*_args, **_kwargs) -> dict:
        raise ModalClientActionError(
            "FileNotFoundError: missing",
            reason="runtime_unavailable",
            error_type="FileNotFoundError",
        )

    monkeypatch.setattr(service, "account_connected", connected)
    monkeypatch.setattr(service, "_run_client_action", missing_remote)

    with pytest.raises(FileNotFoundError, match="동기화되지 않았습니다"):
        await service.start_workflow_run(workflow.name)
