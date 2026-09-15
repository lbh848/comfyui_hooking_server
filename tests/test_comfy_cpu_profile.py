from __future__ import annotations

import copy
import ast
import json
import os
import subprocess
import sys
from pathlib import Path
from threading import Event, RLock
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from comfy_allocation import COMFY_TASK_KEYS
from comfy_installer.e2e import ComfyProcess
from comfy_installer.execution_profile import (
    CPU_LOCAL_TASK_KEYS,
    GENERATION_ONLY_NODE_PACKAGES,
    cpu_launch_args,
    custom_nodes_for_profile,
    installed_cpu_runtime,
    require_local_cpu_task,
)
from comfy_installer.manifest import load_install_manifest
from comfy_installer.model_scope import scope_models, local_model_gaps
from comfy_installer.node_compatibility import validate_instant_lora_export_order
from comfy_installer.service import ComfyInstallerService
from comfy_runtime import ComfyRuntimeManager


def _receipt(root: Path, python: dict) -> None:
    path = root / ".installer-state" / "runtime-receipt.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema_version": 1, "python": python}), encoding="utf-8")


@pytest.mark.parametrize("python,expected", [
    ({"profile_id": "cpu"}, True),
    ({"profile_id": "portable-host", "profile_kind": "cpu"}, True),
    ({"profile_id": "nvidia-cu130"}, False),
])
def test_launch_uses_installed_profile_without_changing_settings(tmp_path, python, expected):
    manager = ComfyRuntimeManager(tmp_path)
    _receipt(manager.comfy_root, python)
    settings = {"cuda_device": 2, "vram_mode": "highvram", "fast": True,
                "extra_args": "--use-sage-attention --fp32-vae"}
    before = copy.deepcopy(settings)
    command, _, _ = manager.build_command(port=8188, profile=settings)
    assert ("--cpu" in command) is expected
    assert ("--highvram" in command) is not expected
    assert ("--cuda-device" in command) is not expected
    assert ("--use-sage-attention" in command) is not expected
    assert "--fp32-vae" in command
    assert settings == before


@pytest.mark.parametrize("cuda,hip,expected", [(None, None, True), ("13.0", None, False), (None, "6.0", False)])
def test_failed_install_without_receipt_reads_target_torch_build(tmp_path, cuda, hip, expected):
    path = tmp_path / ".venv/Lib/site-packages/torch/version.py"
    path.parent.mkdir(parents=True)
    path.write_text(f"cuda: str | None = {cuda!r}\nhip = {hip!r}\n", encoding="utf-8")
    assert installed_cpu_runtime(tmp_path) is expected


@pytest.mark.parametrize("selected_cpu", [True, False])
def test_install_update_launch_uses_new_profile_over_old_receipt(tmp_path, selected_cpu):
    _receipt(tmp_path, {"profile_id": "cpu" if not selected_cpu else "nvidia-cu130"})
    process = ComfyProcess(comfy_root=tmp_path, python=tmp_path / "python.exe",
                           cancel_event=Event(), port=12345, cpu_only=selected_cpu)
    assert ("--cpu" in process.launch_command()) is selected_cpu


def test_cpu_update_preserves_but_does_not_load_old_generation_nodes(tmp_path):
    custom = tmp_path / "custom_nodes"
    for name in ["comfyui-spectrum-ksampler", "shared-utility", "future-pack-node"]:
        (custom / name).mkdir(parents=True)
    args = cpu_launch_args(("--lowvram", "--fast", "fp8_matrix_mult", "--cpu"),
                           cpu_only=True, comfy_root=tmp_path)
    assert args.count("--cpu") == 1
    assert "--lowvram" not in args and "fp8_matrix_mult" not in args
    assert "--disable-all-custom-nodes" in args
    assert "shared-utility" in args and "future-pack-node" in args
    assert "comfyui-spectrum-ksampler" not in args
    assert (custom / "comfyui-spectrum-ksampler").is_dir()


@pytest.mark.parametrize("cpu_only", [True, False])
def test_node_scope_retains_mixed_and_unknown_packages(cpu_only):
    nodes = [{"name": name} for name in [*GENERATION_ONLY_NODE_PACKAGES,
             "comfyui-instant-lora_v_soya", "comfyui-soya-custom-nodes", "new-pack-package"]]
    selected = {n["name"] for n in custom_nodes_for_profile(nodes, cpu_only=cpu_only)}
    assert {"comfyui-instant-lora_v_soya", "comfyui-soya-custom-nodes", "new-pack-package"} <= selected
    assert bool(selected & GENERATION_ONLY_NODE_PACKAGES) is not cpu_only


@pytest.mark.parametrize("generation_binding", ["illustration_workflow_source_paths.v3", "video_workflow_source_paths.i2v", "lora_training_workflow_source_paths.anima"])
def test_cpu_model_scope_follows_semantic_task_binding_and_shared_models(generation_binding):
    # Names and bytes are arbitrary: roles come from bindings, never filenames,
    # catalog model IDs, or an invented model-size threshold.
    workflows = {"items": [
        {"id": "custom-utility", "bindings": ["utility_workflow_source_path"], "model_ids": ["shared", "large-utility"]},
        {"id": "custom-generation", "bindings": [generation_binding], "model_ids": ["shared", "small-generator"]},
    ]}
    models = [{"id": "shared", "size": 3}, {"id": "large-utility", "size": 99_000_000},
              {"id": "small-generator", "size": 1}]
    allocations = {key: 1 for key in COMFY_TASK_KEYS}
    before = copy.deepcopy(allocations)
    scope = scope_models(models, workflows=workflows, allocations=allocations,
                         model_source="local_first", cpu_only=True)
    assert {m["id"] for m in scope.keep} == {"shared", "large-utility"}
    assert {m["id"] for m in scope.skipped} == {"small-generator"}
    assert allocations == before
    allocations["utility_debug"] = "vast"
    assert scope_models(models, workflows=workflows, allocations=allocations,
                        model_source="local_first", cpu_only=True).keep == ()
    assert len(scope_models(models, workflows=workflows, allocations=allocations,
                            model_source="local_first").keep) == 3


def test_cpu_preflight_accepts_space_for_only_lightweight_dependencies(tmp_path, monkeypatch):
    manifest = load_install_manifest()
    service = object.__new__(ComfyInstallerService)
    service.manifest = manifest
    service._read_config = lambda: {"modal_model_source": "local_first"}
    service.preflight = lambda **kwargs: {"gpu_profile": "cpu", "disk": {"free": 45 * 1024**3}}
    probe, scope = service._preflight_models(manifest.models, install_mode="standard", manifest=manifest)
    assert scope.keep and scope.skipped
    assert probe["disk"]["required"] == 30 * 1024**3 + scope.keep_bytes
    assert probe["disk"]["enough"]
    assert "face-yolov8m" in {m["id"] for m in scope.keep}
    assert "anima-base-v1" not in {m["id"] for m in scope.keep}


def test_cpu_e2e_records_generation_as_skipped_not_passed(tmp_path):
    _receipt(tmp_path, {"profile_id": "cpu"})
    service = object.__new__(ComfyInstallerService)
    service.comfy_root = tmp_path
    service._log = lambda *_args: None
    bindings = {"utility_workflow_source_path": "utility.json", "face_extract_workflow_source_path": "face.json",
                "illustration_workflow_source_paths.v3": "image.json", "video_workflow_source_paths.i2v": "video.json",
                "lora_training_workflow_source_paths.anima": "train.json"}
    kept, skipped = service._local_e2e_bindings(bindings)
    assert set(kept.values()) == {"utility.json", "face.json"}
    assert len(skipped) == 3 and all(item["status"] == "skipped" for item in skipped)


def test_cpu_model_diagnostics_do_not_demand_intentionally_omitted_models(tmp_path):
    workflows = {"items": [{"id": "comfy_workflow_source_path", "model_ids": ["gen"]},
                            {"id": "utility_workflow_source_path", "model_ids": ["utility"]}]}
    gaps = local_model_gaps(models=[{"id": "gen", "relative_path": "gen.bin"},
                                   {"id": "utility", "relative_path": "util.bin"}],
                            workflows=workflows, allocations={},
                            config={"comfy_workflow_source_path": "g.json", "utility_workflow_source_path": "u.json"},
                            comfy_root=tmp_path, cpu_only=True)
    assert [m["id"] for m in gaps] == ["utility"]


@pytest.mark.parametrize("task", list(CPU_LOCAL_TASK_KEYS))
def test_cpu_local_utility_is_allowed(tmp_path, task):
    _receipt(tmp_path, {"profile_id": "cpu"})
    require_local_cpu_task(task, comfy_root=tmp_path)


@pytest.mark.parametrize("task", ["illustration", "asset_lora_training", "video_generation"])
def test_cpu_local_generation_fails_before_execution_with_actionable_reason(tmp_path, task):
    _receipt(tmp_path, {"profile_id": "cpu"})
    with pytest.raises(RuntimeError, match="원격 실행 대상"):
        require_local_cpu_task(task, comfy_root=tmp_path)


@pytest.mark.asyncio
@pytest.mark.parametrize("item_type,task,blocked", [
    ("illustration", "illustration", True),
    ("asset_lora_training", "asset_lora_training", True),
    ("data_patch_utility", "utility_debug", False),
    ("tag_analysis", "instance_lora", False),
    ("instance_lora_analysis", "instance_lora", False),
])
async def test_queue_preparation_enforces_cpu_role_but_allows_lora_analysis(tmp_path, item_type, task, blocked):
    _receipt(tmp_path / "comfy", {"profile_id": "cpu"})
    source = (Path(__file__).resolve().parents[1] / "server.py").read_text(encoding="utf-8")
    function = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.AsyncFunctionDef) and node.name == "prepare_local_gpu_execution_target")
    engine = SimpleNamespace(ensure_cold_for_comfy=AsyncMock(return_value={}))
    namespace = {"os": os, "BASE_DIR": str(tmp_path), "VIDEO_ENGINE_TARGET": "video_engine",
                 "video_engine_service": engine,
                 "queue_manager": SimpleNamespace(_comfy_task_key_for_item=lambda item: task)}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "server.py", "exec"), namespace)
    call = namespace[function.name]("local", SimpleNamespace(id="test-job", type=item_type))
    if blocked:
        with pytest.raises(RuntimeError, match="CPU 경량 런타임"):
            await call
    else:
        await call
    engine.ensure_cold_for_comfy.assert_not_awaited()


def test_order_probe_enables_cpu_before_importing_node(tmp_path, monkeypatch):
    root = tmp_path / "comfy"
    comfy = root / "comfy"
    src = root / "custom_nodes/comfyui-instant-lora_v_soya/src"
    comfy.mkdir(parents=True)
    src.mkdir(parents=True)
    (comfy / "__init__.py").write_text("", encoding="utf-8")
    (comfy / "options.py").write_text("args_parsing=False\ndef enable_args_parsing():\n    global args_parsing\n    args_parsing=True\n", encoding="utf-8")
    (comfy / "model_management.py").write_text(
        "import sys\nfrom comfy.options import args_parsing\n"
        "if not args_parsing or '--cpu' not in sys.argv:\n"
        "    raise AssertionError('Torch not compiled with CUDA enabled')\n", encoding="utf-8")
    (src / "__init__.py").write_text("from . import nodes\n", encoding="utf-8")
    (src / "nodes.py").write_text(
        "import comfy.model_management\nfrom pathlib import Path\n"
        "class ContextBuilderPathOnlyV1:\n"
        "    def build(self,prompt,negative,root):\n"
        "        entries=[]\n"
        "        for path,line in zip(sorted(Path(root).glob('*.png')),prompt.splitlines()):\n"
        "            caption=line.split(']',1)[1]\n"
        "            path.with_suffix('.txt').write_text(caption,encoding='utf-8')\n"
        "            entries.append({'positive_tags':caption})\n"
        "        return ({'entries':entries,'image_count':len(entries)},)\n", encoding="utf-8")
    def run(command, **kwargs):
        result = subprocess.run(command, cwd=kwargs["cwd"], capture_output=True, text=True, encoding="utf-8", timeout=30)
        assert result.returncode == 0, result.stderr
        return result.stdout.splitlines()
    monkeypatch.setattr("comfy_installer.node_compatibility.run_command", run)
    result = validate_instant_lora_export_order(comfy_root=root, python=Path(sys.executable), cancel_event=Event())
    assert result["image_count"] == 12 and result["status"] == "success"
    assert not list((root / ".installer-state/e2e").glob("lora-export-order-*"))


@pytest.mark.parametrize("kind", ["cpu", "nvidia"])
def test_install_pipeline_uses_selected_profile_for_models_nodes_and_startup(tmp_path, monkeypatch, kind):
    import comfy_installer.service as module
    manifest = load_install_manifest()
    profile_id = "cpu" if kind == "cpu" else "nvidia-cu130"
    service = object.__new__(ComfyInstallerService)
    service.project_root = tmp_path
    service.comfy_root = tmp_path / "comfy"
    service.workflow_library_root = tmp_path / "library"
    service.runtime_backup_dir = tmp_path / "backups"
    service.config_backup_dir = tmp_path / "backups"
    service.config_path = tmp_path / "config.json"
    service.manifest = manifest
    service.downloader = None
    service._cancel = Event()
    service._lock = RLock()
    service._state = {}
    service._read_config = lambda: {"modal_model_source": "local_first"}
    service._log = lambda *_args: None
    service._log_comfy = lambda *_args: None
    service._set_phase = lambda *_args: None
    service._write_result = lambda result: tmp_path / "result.json"
    service._embedded_workflow_base_dir = lambda: str(tmp_path / "workflows")
    service.get_civitai_key = lambda: ""
    service._validate_civitai_access = lambda *_args: None
    service.preflight = lambda **_kwargs: {"gpu_profile": profile_id, "disk": {"free": 500 * 1024**3}}
    selection = SimpleNamespace(release_version="v4", selected_item_ids=("all",),
        model_ids=tuple(m["id"] for m in manifest.models), workflow_bindings={}, user_files=())
    captured = {}
    monkeypatch.setattr(module, "release_install_manifest", lambda **_kwargs: manifest)
    monkeypatch.setattr(module, "selection_requirements", lambda **_kwargs: {"model_ids": selection.model_ids})
    monkeypatch.setattr(module, "import_user_copies", lambda **_kwargs: selection)
    monkeypatch.setattr(module, "install_comfy_source", lambda **_kwargs: None)
    monkeypatch.setattr(module, "create_comfy_venv", lambda **_kwargs: tmp_path / "python.exe")
    monkeypatch.setattr(module, "install_python_dependencies", lambda **kwargs: {"profile": kwargs["gpu_profile"]["id"]})
    monkeypatch.setattr(module, "install_manager_dependencies", lambda **_kwargs: {})
    def install_nodes(**kwargs):
        captured["nodes"] = kwargs["nodes"]
        return [service.comfy_root / "custom_nodes" / node["name"] for node in kwargs["nodes"]]
    monkeypatch.setattr(module, "install_custom_nodes", install_nodes)
    monkeypatch.setattr(module, "install_node_dependencies", lambda **_kwargs: [])
    monkeypatch.setattr(module, "verify_isolated_runtime", lambda **kwargs: {"kind": kwargs["gpu_profile"]["kind"]})
    monkeypatch.setattr(module, "validate_instant_lora_export_order", lambda **_kwargs: {"status": "success", "image_count": 12})
    def install_models(**kwargs):
        captured["models"] = kwargs["models"]
        return []
    monkeypatch.setattr(module, "install_models", install_models)
    monkeypatch.setattr(module, "patch_comfy_input", lambda **_kwargs: {})
    monkeypatch.setattr(module, "git_head", lambda *_args: "a" * 40)
    monkeypatch.setattr(module, "apply_installed_config", lambda **_kwargs: SimpleNamespace(
        backup_path=tmp_path / "backup.json", before_sha256="a", after_sha256="b"))
    monkeypatch.setattr(module, "write_runtime_receipt", lambda **kwargs: {"profile_id": kwargs["profile_id"]})
    class Process(ComfyProcess):
        def start(self, **_kwargs):
            captured["command"] = self.launch_command()
            return {"system": {"comfyui_version": "fixture"}}
        def stop(self):
            captured["stopped"] = True
    monkeypatch.setattr(module, "ComfyProcess", Process)
    service._run_install(release_version="v4", selected_item_ids=["all"], install_mode="standard")
    assert service._state["state"] == "succeeded", service._state
    assert captured["stopped"]
    assert ("--cpu" in captured["command"]) is (kind == "cpu")
    assert ("anima-base-v1" in {m["id"] for m in captured["models"]}) is (kind == "nvidia")
    assert ("comfyui-spectrum-ksampler" in {n["name"] for n in captured["nodes"]}) is (kind == "nvidia")
    assert "face-yolov8m" in {m["id"] for m in captured["models"]}
    assert not service.config_path.exists()
