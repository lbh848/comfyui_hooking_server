from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest

import comfy_installer.service as service_module
from comfy_installer.e2e import ComfyE2EError
from comfy_installer.manifest import load_install_manifest
from comfy_installer.service import ComfyInstallerService


def test_service_status_never_contains_credentials(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    config.write_text("{}\n", encoding="utf-8")
    service = ComfyInstallerService(
        project_root=tmp_path,
        config_path=config,
        requirements_dir=tmp_path / "requirements",
    )
    payload = json.dumps(service.status(), ensure_ascii=False)
    assert "civitai_key" not in payload
    assert "workflow_key" not in payload
    assert service.comfy_root == tmp_path / "comfy"


def test_installed_compatibility_mode_is_reused_for_updates(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    config.write_text("{}\n", encoding="utf-8")
    service = ComfyInstallerService(
        project_root=tmp_path,
        config_path=config,
        requirements_dir=tmp_path / "requirements",
    )
    state_root = tmp_path / "comfy" / ".installer-state"
    state_root.mkdir(parents=True)
    (state_root / "install-result-20260730_120000_000001.json").write_text(
        json.dumps(
            {
                "operation": "install",
                "install_mode": "nvidia_compatibility",
                "python": {"profile": "nvidia-cu128"},
            }
        ),
        encoding="utf-8",
    )

    assert service._installed_install_mode() == "nvidia_compatibility"


def test_v4_migration_backs_up_copies_and_retargets_config(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    requirements = tmp_path / "요구사항"
    old_comfy = tmp_path / "old-comfy"
    embedded_comfy = tmp_path / "comfy"
    old_lora = (
        old_comfy
        / "models"
        / "loras"
        / "SOYA_CHAR_LORA"
        / "character.safetensors"
    )
    old_lora.parent.mkdir(parents=True)
    old_lora.write_bytes(b"lora")
    (embedded_comfy / ".git").mkdir(parents=True)
    original = {
        "comfy_input_dir": str(old_comfy / "input"),
        "nested": {
            "lora": str(
                old_comfy / "models" / "loras" / "SOYA_CHAR_LORA"
            )
        },
    }
    config.write_text(
        json.dumps(original, ensure_ascii=False),
        encoding="utf-8",
    )
    service = ComfyInstallerService(
        project_root=tmp_path,
        config_path=config,
        requirements_dir=requirements,
    )

    result = service.migrate_from_existing_comfy(old_comfy)

    backup = Path(result["config"]["backup_path"])
    assert backup.parent == requirements
    assert json.loads(backup.read_text(encoding="utf-8")) == original
    updated = json.loads(config.read_text(encoding="utf-8"))
    assert updated["comfy_input_dir"] == str(embedded_comfy / "input")
    assert updated["nested"]["lora"] == str(
        embedded_comfy / "models" / "loras" / "SOYA_CHAR_LORA"
    )
    assert (
        embedded_comfy
        / "models"
        / "loras"
        / "SOYA_CHAR_LORA"
        / "character.safetensors"
    ).read_bytes() == b"lora"
    assert result["config"]["updated_paths"] == [
        "$.comfy_input_dir",
        "$.nested.lora",
    ]
    assert result["config"]["already_retargeted"] is False

    repeated = service.migrate_from_existing_comfy(old_comfy)

    assert repeated["copied"] == []
    assert len(repeated["skipped"]) == 1
    assert repeated["config"]["updated_paths"] == []
    assert repeated["config"]["already_retargeted"] is True


def test_v4_migration_runs_in_background_and_publishes_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.json"
    config.write_text("{}\n", encoding="utf-8")
    service = ComfyInstallerService(
        project_root=tmp_path,
        config_path=config,
        requirements_dir=tmp_path / "requirements",
    )
    worker_started = Event()
    allow_finish = Event()

    def fake_migration(_old_comfy_root, **kwargs):
        kwargs["set_phase"]("migration_scan")
        kwargs["progress"](
            {
                "event": "migration_copy",
                "engine": "robocopy",
                "current": 1,
                "total": 2,
                "overall_downloaded": 10,
                "overall_total": 20,
                "bytes_per_second": 5,
                "eta_seconds": 2,
            }
        )
        worker_started.set()
        assert allow_finish.wait(timeout=5)
        return {
            "copy_engine": "robocopy",
            "pending_bytes": 20,
            "copied": ["one", "two"],
            "skipped": ["existing"],
            "missing": [],
            "failures": [],
            "config": {
                "updated_paths": [],
                "missing_targets": [],
                "already_retargeted": True,
            },
        }

    monkeypatch.setattr(service, "_perform_migration", fake_migration)

    started = service.start_migration(str(tmp_path / "old-comfy"))
    assert worker_started.wait(timeout=5)
    running = service.status()
    assert started["operation"] == "migrate"
    assert running["state"] == "running"
    assert running["progress"]["engine"] == "robocopy"
    assert running["progress"]["eta_seconds"] == 2

    allow_finish.set()
    assert service._thread is not None
    service._thread.join(timeout=5)
    finished = service.status()
    assert finished["state"] == "succeeded"
    assert finished["operation"] == "migrate"
    assert finished["result"]["copy_engine"] == "robocopy"
    assert finished["result"]["copied"] == ["one", "two"]


def test_comfy_warning_is_visible_in_status_without_console_echo(
    tmp_path: Path,
    capsys,
) -> None:
    config = tmp_path / "config.json"
    config.write_text("{}\n", encoding="utf-8")
    service = ComfyInstallerService(
        project_root=tmp_path,
        config_path=config,
        requirements_dir=tmp_path / "requirements",
    )

    service._log_comfy(
        "[Comfy][WARNING] 'lora key not loaded' 경고 누적 100건"
    )

    assert capsys.readouterr().out == ""
    entry = service.status()["logs"][-1]
    assert entry["level"] == "warning"
    assert "누적 100건" in entry["message"]


def test_runtime_e2e_records_failure_and_continues_remaining_workflows(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = tmp_path / "config.json"
    config.write_text("{}\n", encoding="utf-8")
    service = ComfyInstallerService(
        project_root=tmp_path,
        config_path=config,
        requirements_dir=tmp_path / "requirements",
    )
    validations = [
        SimpleNamespace(
            binding_keys=(f"fixture.{name}",),
            filename=f"{name}.json",
            prompt={},
            workflow={"nodes": []},
        )
        for name in ("a", "b", "c")
    ]
    calls: list[str] = []

    @contextmanager
    def fake_fixtures(**_kwargs):
        yield {"training": "fixture/training/sample.png"}

    def fake_execute_prompt(**kwargs):
        filename = kwargs["filename"]
        calls.append(filename)
        if filename == "a.json":
            raise ComfyE2EError("fixture execution error")
        return {
            "filename": filename,
            "prompt_id": f"prompt-{filename}",
            "status": "success",
            "outputs": [],
            "output_data": {},
        }

    monkeypatch.setattr(
        service_module,
        "protected_e2e_fixtures",
        fake_fixtures,
    )
    monkeypatch.setattr(
        service_module,
        "make_e2e_prompt",
        lambda _validation: {},
    )
    monkeypatch.setattr(
        service_module,
        "execute_prompt",
        fake_execute_prompt,
    )
    monkeypatch.setattr(
        service_module,
        "promote_generated_fixture",
        lambda **_kwargs: None,
    )
    process = SimpleNamespace(
        base_url="http://127.0.0.1:12345",
        process=SimpleNamespace(poll=lambda: None),
    )

    with pytest.raises(ComfyE2EError, match="성공 2/3, 실패 1/3"):
        service._run_runtime_e2e(
            process=process,
            validations=validations,
        )

    assert calls == ["a.json", "b.json", "c.json"]
    progress = service.status()["progress"]
    assert progress["current"] == 3
    assert progress["succeeded"] == 2
    assert progress["failed"] == 1
    assert progress["failed_filenames"] == ["a.json"]


def test_manifest_has_fully_pinned_windows_runtime_and_assets() -> None:
    manifest = load_install_manifest()
    assert manifest.comfy["version"] == "0.20.1"
    assert len(manifest.comfy["ref"]) == 40
    assert manifest.python["version"] == "3.12.11"
    assert manifest.python["compatibility_packages"] == [
        "numpy==1.26.4",
        "scipy==1.14.1",
        "tifffile==2024.9.20",
        "opencv-python==4.10.0.84",
        "opencv-python-headless==4.10.0.84",
        "opencv-contrib-python==4.10.0.84",
    ]
    nvidia = {
        profile["id"]: profile
        for profile in manifest.python["gpu_profiles"]
        if profile["kind"] == "nvidia"
    }
    assert set(nvidia) == {"nvidia-cu128", "nvidia-cu130"}
    assert nvidia["nvidia-cu128"]["minimum_driver_version"] == "570.65"
    assert nvidia["nvidia-cu128"]["minimum_compute_capability"] == "8.0"
    assert nvidia["nvidia-cu128"]["torch_cuda"] == "12.8"
    assert nvidia["nvidia-cu128"]["sageattention"]["size"] == 12252026
    assert nvidia["nvidia-cu128"]["sageattention"]["sha256"] == (
        "b8b3134d00dfbdae5c10cc34cc8508891d9420adaa182502fa30a496428531ed"
    )
    assert nvidia["nvidia-cu130"]["minimum_driver_version"] == "580.00"
    assert nvidia["nvidia-cu130"]["minimum_compute_capability"] == "8.0"
    assert nvidia["nvidia-cu130"]["torch_cuda"] == "13.0"
    assert nvidia["nvidia-cu130"]["sageattention"]["size"] == 12321863
    assert all(
        len(profile["sageattention"]["sha256"]) == 64
        for profile in nvidia.values()
    )
    assert len(manifest.custom_nodes) == 15
    lora_manager = next(
        node
        for node in manifest.custom_nodes
        if node["name"] == "comfyui-lora-manager"
    )
    assert lora_manager["repository"] == (
        "https://github.com/willmiao/ComfyUI-Lora-Manager.git"
    )
    assert lora_manager["ref"] == (
        "0d8805cdee93d1a7347a813f4ec271ba6bcb55f5"
    )
    spectrum = next(
        node
        for node in manifest.custom_nodes
        if node["name"] == "comfyui-spectrum-ksampler"
    )
    assert spectrum["ref"] == "c806917566ee1c149575cd90da9e4c2e543de019"
    tracking_main_names = {
        node["name"]
        for node in manifest.custom_nodes
        if node.get("tracking_branch") == "main"
    }
    assert tracking_main_names == {
        "comfyui-instant-lora_v_soya",
        "comfyui-soya-custom-nodes",
        "comfyui-workflow-to-api-converter-endpoint",
    }
    assert all(
        "ref" not in node
        for node in manifest.custom_nodes
        if node["name"] in tracking_main_names
    )
    assert len(manifest.models) == 36
    assert manifest.workflows["expected_count"] == 17
    fixed_v1 = manifest.workflows["release_dependencies"]["v1"]
    assert len(fixed_v1) == 17
    assert {
        binding
        for item in fixed_v1
        for binding in item["bindings"]
    } == set(manifest.workflows["required_bindings"])
    qwen = next(
        item for item in fixed_v1
        if item["id"] == "qwen_edit_workflow_source_path"
    )
    assert qwen["model_ids"] == ["qwen-image-edit-rapid-v19"]
    assert "캐릭터복장추적_v1.json" in manifest.workflows[
        "excluded_filenames"
    ]
