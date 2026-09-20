from __future__ import annotations

import ast
import copy
import importlib.util
import inspect
import json
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest

import comfy_installer.image_diagnostic as image_diagnostic_module
from comfy_installer.e2e import WorkflowValidation
from comfy_installer.image_diagnostic import (
    _profile_with,
    assess_run,
    prepare_diagnostic_prompt,
)
from comfy_installer.service import ComfyInstallerService, InstallerServiceError


def _validation(prompt: dict) -> WorkflowValidation:
    return WorkflowValidation(
        binding_keys=("anima_only_asset_workflow_source_path",),
        filename="asset.json",
        node_count=len(prompt),
        class_count=len({node["class_type"] for node in prompt.values()}),
        classes=tuple(sorted({node["class_type"] for node in prompt.values()})),
        prompt=prompt,
        workflow={"nodes": []},
    )


def _ksampler_prompt() -> dict:
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "anima.safetensors"}},
        "2": {"class_type": "DCWModelPatch", "inputs": {"model": ["1", 0]}},
        "3": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": 1408, "height": 2048, "batch_size": 4},
        },
        "4": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["2", 0],
                "seed": 777,
                "steps": 30,
                "cfg": 5.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "positive": ["20", 0],
                "negative": ["21", 0],
                "latent_image": ["3", 0],
                "denoise": 1.0,
            },
        },
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": "wan_vae.safetensors"}},
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["4", 0], "vae": ["5", 0]}},
        "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": "user"}},
        "8": {
            "class_type": "PrimitiveStringMultiline",
            "inputs": {
                "value": "[LORA_ACTIVATE]\ntrue\n[WIDTH]\n1408\n[HEIGHT]\n2048\n[N_IMG]\n4\n"
            },
        },
        "20": {"class_type": "Conditioning", "inputs": {"config": ["8", 0]}},
        "21": {"class_type": "Conditioning", "inputs": {}},
    }


def test_prepare_diagnostic_prompt_preserves_real_steps_and_resolution(tmp_path: Path) -> None:
    original = _ksampler_prompt()
    validation = _validation(copy.deepcopy(original))
    prepared = prepare_diagnostic_prompt(
        validation,
        output_dir=tmp_path,
        profile_name="current",
    )

    prompt = prepared["prompt"]
    assert prompt["4"]["inputs"]["steps"] == 30
    assert prompt["3"]["inputs"]["width"] == 1408
    assert prompt["3"]["inputs"]["height"] == 2048
    assert prompt["3"]["inputs"]["batch_size"] == 1
    assert "[LORA_ACTIVATE]\nfalse" in prompt["8"]["inputs"]["value"]
    assert "[WIDTH]\n1408" in prompt["8"]["inputs"]["value"]
    assert "[N_IMG]\n1" in prompt["8"]["inputs"]["value"]
    assert "7" not in prompt
    assert prompt["4"]["inputs"]["model"] == ["__lb_diag_model_probe", 0]
    assert prompt["4"]["inputs"]["latent_image"] == ["__lb_diag_latent_trigger", 0]
    assert prompt["6"]["inputs"]["samples"] == ["__lb_diag_latent_probe", 0]
    assert prepared["sampler"]["class_type"] == "KSampler"
    assert original["3"]["inputs"]["batch_size"] == 4


def test_unpatched_prompt_bypasses_model_and_clip_outputs(tmp_path: Path) -> None:
    prompt = _ksampler_prompt()
    prompt["2"] = {
        "class_type": "Power Lora Loader (rgthree)",
        "inputs": {"model": ["1", 0], "clip": ["1", 1]},
    }
    prompt["20"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 1], "text": "test"}}
    validation = _validation(prompt)

    prepared = prepare_diagnostic_prompt(
        validation,
        output_dir=tmp_path,
        profile_name="core_unpatched",
        unpatched=True,
    )

    result = prepared["prompt"]
    assert "2" not in result
    assert result["__lb_diag_model_probe"]["inputs"]["model"] == ["1", 0]
    assert result["20"]["inputs"]["clip"] == ["1", 1]
    assert prepared["bypassed"] == [
        {"node_id": "2", "class_type": "Power Lora Loader (rgthree)"}
    ]


def test_unpatched_prompt_resolves_chained_patch_links(tmp_path: Path) -> None:
    prompt = _ksampler_prompt()
    prompt["9"] = {
        "class_type": "PathchSageAttentionKJ",
        "inputs": {"model": ["1", 0]},
    }
    prompt["2"] = {
        "class_type": "Power Lora Loader (rgthree)",
        "inputs": {"model": ["9", 0], "clip": ["1", 1]},
    }
    prepared = prepare_diagnostic_prompt(
        _validation(prompt),
        output_dir=tmp_path,
        profile_name="core_unpatched",
        unpatched=True,
    )
    result = prepared["prompt"]
    assert "2" not in result
    assert "9" not in result
    assert result["__lb_diag_model_probe"]["inputs"]["model"] == ["1", 0]


def test_unpatched_first_sampler_becomes_builtin_ksampler(tmp_path: Path) -> None:
    prompt = _ksampler_prompt()
    prompt["4"]["class_type"] = "SoyaFirstSampler_mdsoya"
    prompt["4"]["inputs"].update(
        {
            "sampler_mode": "SpectrumKSamplerModGuidance",
            "LORA_ACT": "false",
            "LORA_DATA": '{"list":[]}',
            "multi_char": "false",
        }
    )
    prepared = prepare_diagnostic_prompt(
        _validation(prompt),
        output_dir=tmp_path,
        profile_name="core_unpatched",
        unpatched=True,
    )
    assert prepared["sampler"]["class_type"] == "KSampler"
    assert "sampler_mode" not in prepared["prompt"]["4"]["inputs"]


def test_assess_run_distinguishes_model_latent_and_vae_failures() -> None:
    base_image = {
        "finite": True,
        "black_fraction": 0.0,
        "luminance_std": 0.2,
        "edge_mean": 0.05,
        "thumbnail_16x16_rgb": [0.2, 0.4, 0.6],
    }
    reference = {"image": base_image, "model_calls": []}

    model_bad = {
        "model_calls": [{"output": {"finite": False}}],
        "latent": {"finite": False},
        "image": {**base_image, "finite": False},
    }
    assert assess_run(model_bad, reference)["stage"] == "diffusion_model"

    latent_bad = {
        "model_calls": [{"output": {"finite": True}}],
        "latent": {"finite": False},
        "image": {**base_image, "finite": False},
    }
    assert assess_run(latent_bad, reference)["stage"] == "sampler_latent"

    vae_bad = {
        "model_calls": [{"output": {"finite": True}}],
        "latent": {"finite": True},
        "image": {**base_image, "finite": False},
    }
    assert assess_run(vae_bad, reference)["stage"] == "vae_decode"


def test_assess_run_marks_same_seed_visual_drift_before_nan() -> None:
    reference = {
        "image": {
            "finite": True,
            "black_fraction": 0.0,
            "luminance_std": 0.2,
            "edge_mean": 0.04,
            "thumbnail_16x16_rgb": [0.1, 0.2, 0.3],
        }
    }
    drift = {
        "model_calls": [{"output": {"finite": True}}],
        "latent": {"finite": True},
        "image": {
            "finite": True,
            "black_fraction": 0.0,
            "luminance_std": 0.25,
            "edge_mean": 0.3,
            "thumbnail_16x16_rgb": [0.8, 0.7, 0.6],
        },
    }
    assessment = assess_run(drift, reference)
    assert assessment["abnormal"] is True
    assert assessment["stage"] is None
    assert any("동일 seed" in reason for reason in assessment["reasons"])
    assert any("컬러 노이즈" in reason for reason in assessment["reasons"])


def test_assess_run_marks_finite_model_drift_as_precursor() -> None:
    image = {
        "finite": True,
        "black_fraction": 0.0,
        "luminance_std": 0.2,
        "edge_mean": 0.04,
        "thumbnail_16x16_rgb": [0.1, 0.2, 0.3],
    }
    reference = {
        "model_calls": [
            {"output": {"finite": True, "sample_preview": [0.0, 1.0], "std": 0.5}}
        ],
        "latent": {"finite": True, "sample_preview": [0.0, 1.0], "std": 0.5},
        "image": image,
    }
    drift = {
        "model_calls": [
            {"output": {"finite": True, "sample_preview": [0.5, 1.5], "std": 0.5}}
        ],
        "latent": {"finite": True, "sample_preview": [0.5, 1.5], "std": 0.5},
        "image": image,
    }
    assessment = assess_run(drift, reference)
    assert assessment["abnormal"] is True
    assert assessment["model_drift_call"] == 0
    assert assessment["model_relative_mae"] == pytest.approx(1.0)
    assert assessment["latent_relative_mae"] == pytest.approx(1.0)


def test_temporary_probe_resource_registers_all_boundary_nodes() -> None:
    root = Path(__file__).resolve().parents[1]
    resource = root / "comfy_installer" / "resources" / "image_diagnostic_node"
    marker = (resource / ".comfy-installer-owned").read_text(encoding="utf-8").strip()
    source = (resource / "__init__.py").read_text(encoding="utf-8")
    assert marker == "comfy-installer-image-diagnostic-v1"
    for class_name in (
        "LBDiagnosticModelProbe",
        "LBDiagnosticLatentTrigger",
        "LBDiagnosticLatentProbe",
        "LBDiagnosticImageProbe",
    ):
        assert f'"{class_name}"' in source
    assert "NODE_CLASS_MAPPINGS" in source


def test_temporary_probe_measures_fake_model_latent_and_image(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    root = Path(__file__).resolve().parents[1]
    source = root / "comfy_installer" / "resources" / "image_diagnostic_node" / "__init__.py"
    spec = importlib.util.spec_from_file_location("test_lb_image_diagnostic_node", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class FakeModel:
        def __init__(self) -> None:
            self.model_options = {}
            self.model = SimpleNamespace(
                diffusion_model=torch.nn.Sequential(torch.nn.Linear(2, 2))
            )

        def clone(self):
            cloned = FakeModel()
            cloned.model = self.model
            cloned.model_options = dict(self.model_options)
            return cloned

        def set_model_unet_function_wrapper(self, wrapper) -> None:
            self.model_options["model_function_wrapper"] = wrapper

    run_id = "unit-run"
    patched = module.LBDiagnosticModelProbe().probe(FakeModel(), run_id, -1)[0]
    wrapper = patched.model_options["model_function_wrapper"]
    value = torch.tensor([[1.0, 2.0]])
    output = wrapper(
        lambda input_value, _timestep, **_kwargs: input_value * 2,
        {"input": value, "timestep": torch.tensor([1.0]), "c": {}},
    )
    assert torch.equal(output, value * 2)

    latent = {"samples": output.reshape(1, 2, 1, 1)}
    assert module.LBDiagnosticLatentProbe().probe(latent, run_id)[0] is latent
    image = torch.full((1, 8, 8, 3), 0.25)
    response = module.LBDiagnosticImageProbe().measure(
        image, run_id, "unit", str(tmp_path)
    )
    payload = json.loads(response["ui"]["diagnostic"][0])
    assert payload["model_calls"][0]["output"]["finite"] is True
    assert payload["latent"]["finite"] is True
    assert payload["image"]["finite"] is True
    assert (tmp_path / payload["artifact"]).is_file()


def test_temporary_probe_deep_trace_finds_finite_to_nonfinite_leaf(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    root = Path(__file__).resolve().parents[1]
    source = root / "comfy_installer" / "resources" / "image_diagnostic_node" / "__init__.py"
    spec = importlib.util.spec_from_file_location("test_lb_image_diagnostic_deep", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class FakeModel:
        def __init__(self) -> None:
            self.model_options = {}
            self.model = SimpleNamespace(
                diffusion_model=torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False))
            )

        def clone(self):
            cloned = FakeModel()
            cloned.model = self.model
            cloned.model_options = dict(self.model_options)
            return cloned

        def set_model_unet_function_wrapper(self, wrapper) -> None:
            self.model_options["model_function_wrapper"] = wrapper

    run_id = "deep-run"
    patched = module.LBDiagnosticModelProbe().probe(FakeModel(), run_id, 0)[0]
    with torch.no_grad():
        patched.model.diffusion_model[0].weight.fill_(float("inf"))
    wrapper = patched.model_options["model_function_wrapper"]
    value = torch.tensor([[1.0, 2.0]])
    output = wrapper(
        lambda input_value, _timestep, **_kwargs: patched.model.diffusion_model(input_value),
        {"input": value, "timestep": torch.tensor([1.0]), "c": {}},
    )
    latent = {"samples": output.reshape(1, 2, 1, 1)}
    module.LBDiagnosticLatentProbe().probe(latent, run_id)
    response = module.LBDiagnosticImageProbe().measure(
        torch.zeros((1, 4, 4, 3)), run_id, "deep", str(tmp_path)
    )
    payload = json.loads(response["ui"]["diagnostic"][0])
    assert payload["model_calls"][0]["output"]["finite"] is False
    assert payload["deep_transitions"]
    assert payload["deep_transitions"][0]["input"]["finite"] is True
    assert payload["deep_transitions"][0]["output"]["finite"] is False


def test_diagnostic_archive_lookup_is_confined_to_work_root(tmp_path: Path) -> None:
    service = object.__new__(ComfyInstallerService)
    service.project_root = tmp_path
    diagnostic_id = "20260920_120000-1234abcd"
    archive = (
        tmp_path / ".work" / "comfy-installer" / "image-diagnostics" / f"{diagnostic_id}.zip"
    )
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"zip")
    assert service.image_diagnostic_archive(diagnostic_id) == archive.resolve()
    with pytest.raises(InstallerServiceError):
        service.image_diagnostic_archive("../../config")


def test_comparison_profile_can_force_gpu_or_cpu_vae() -> None:
    base = {
        "vram_mode": "auto",
        "disable_dynamic_vram": False,
        "extra_args": "--cpu-vae --preview-method none",
    }
    gpu_vae = _profile_with(
        base, vram_mode="novram", dynamic_off=True, cpu_vae=False
    )
    assert gpu_vae["vram_mode"] == "novram"
    assert gpu_vae["disable_dynamic_vram"] is True
    assert "--cpu-vae" not in gpu_vae["extra_args"]
    assert "--preview-method none" in gpu_vae["extra_args"]

    cpu_vae = _profile_with(
        base, vram_mode="highvram", dynamic_off=True, cpu_vae=True
    )
    assert cpu_vae["vram_mode"] == "highvram"
    assert cpu_vae["extra_args"].count("--cpu-vae") == 1


def test_every_diagnostic_comfy_process_does_not_load_or_verify_manager() -> None:
    tree = ast.parse(inspect.getsource(image_diagnostic_module))
    process_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ComfyProcess"
    ]
    assert len(process_calls) == 2
    for call in process_calls:
        verify_manager = next(
            (keyword.value for keyword in call.keywords if keyword.arg == "verify_manager"),
            None,
        )
        enable_manager = next(
            (keyword.value for keyword in call.keywords if keyword.arg == "enable_manager"),
            None,
        )
        assert isinstance(verify_manager, ast.Constant)
        assert verify_manager.value is False
        assert isinstance(enable_manager, ast.Constant)
        assert enable_manager.value is False


def test_diagnostic_conversion_posts_clean_workflow_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workflow_path = tmp_path / "clean-pack-workflow.json"
    workflow_path.write_text(
        json.dumps({"nodes": [{"id": 1, "type": "KSampler"}]}),
        encoding="utf-8",
    )
    calls: list[tuple[str, dict]] = []

    class FakeResponse:
        status_code = 200
        text = ""

        @staticmethod
        def json() -> dict:
            return {
                "prompt": {
                    "1": {"class_type": "KSampler", "inputs": {}},
                }
            }

    class FakeClient:
        def __init__(self, **_kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def post(self, path: str, *, json: dict):
            calls.append((path, json))
            return FakeResponse()

    monkeypatch.setattr(image_diagnostic_module.httpx, "Client", FakeClient)
    validation = image_diagnostic_module._convert_workflow_once(
        base_url="http://127.0.0.1:12345",
        workflow_path=workflow_path,
        binding="anima_only_asset_workflow_source_path",
        cancel_event=Event(),
    )

    assert [path for path, _ in calls] == ["/workflow/convert"]
    assert validation.filename == workflow_path.name
    assert validation.binding_keys == ("anima_only_asset_workflow_source_path",)


def test_diagnostic_sanitizing_never_replaces_user_or_face_inputs() -> None:
    source = {
        "1": {
            "class_type": "LoadImage",
            "inputs": {"image": "pack-default.webp"},
        },
        "2": {
            "class_type": "PrimitiveStringMultiline",
            "inputs": {
                "value": (
                    "[FACE_ID_ACTIVATE]\ntrue\n"
                    "[FACE_ID_DIR]\nsoya_char_ref/fallback\n"
                    "[STYLE_DIR]\nsoya_style_ref/fallback\n"
                    "[LORA_ACTIVATE]\ntrue\n"
                )
            },
        },
    }

    sanitized = image_diagnostic_module._sanitize_prompt(source)

    assert sanitized["1"]["inputs"]["image"] == "pack-default.webp"
    value = sanitized["2"]["inputs"]["value"]
    assert "[FACE_ID_ACTIVATE]\nfalse" in value
    assert "[LORA_ACTIVATE]\nfalse" in value
    assert "[FACE_ID_DIR]\nsoya_char_ref/fallback" in value
    assert "[STYLE_DIR]\nsoya_style_ref/fallback" in value
    assert "comfy-installer-e2e" not in value


def test_diagnostic_uses_pack_distribution_original_not_configured_user_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clean = tmp_path / "SOYA_DISTRIBUTION" / "v4" / "clean.json"
    clean.parent.mkdir(parents=True)
    clean.write_text("{}", encoding="utf-8")
    captured: dict = {}

    def fake_resolve_distribution_selection(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            workflow_bindings={
                "anima_only_asset_workflow_source_path": str(clean)
            }
        )

    monkeypatch.setattr(
        image_diagnostic_module,
        "resolve_distribution_selection",
        fake_resolve_distribution_selection,
    )
    binding, path, workflow_type = image_diagnostic_module._workflow_binding(
        {
            "asset_workflow_type": "anima_only",
            "anima_only_asset_workflow_source_path": str(
                tmp_path / "SOYA_USER" / "modified.json"
            ),
        },
        workflow_library_root=tmp_path,
        workflow_release="v4",
    )

    assert binding == "anima_only_asset_workflow_source_path"
    assert path == clean.resolve()
    assert workflow_type == "anima_only"
    assert captured["release_version"] == "v4"
    assert captured["selected_item_ids"] == [binding]
