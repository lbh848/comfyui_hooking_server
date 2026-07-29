from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
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
    nvidia = next(
        profile
        for profile in manifest.python["gpu_profiles"]
        if profile["kind"] == "nvidia"
    )
    assert nvidia["id"] == "nvidia-cu130"
    assert nvidia["sageattention"]["size"] == 12321863
    assert len(nvidia["sageattention"]["sha256"]) == 64
    assert len(manifest.custom_nodes) == 14
    spectrum = next(
        node
        for node in manifest.custom_nodes
        if node["name"] == "comfyui-spectrum-ksampler"
    )
    assert spectrum["ref"] == "c806917566ee1c149575cd90da9e4c2e543de019"
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
