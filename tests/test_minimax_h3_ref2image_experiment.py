import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = ROOT / "experiments" / "minimax_h3_ref2image"
MODULE_PATH = EXPERIMENT_DIR / "run_probe.py"
SPEC = importlib.util.spec_from_file_location("minimax_h3_ref2image_probe", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _args(**overrides):
    values = {
        "comfy_url": "http://127.0.0.1:8188",
        "references": ["reference.png"],
        "prompt": "Use <Picture 1> as the identity reference.",
        "width": 1024,
        "height": 768,
        "steps": 12,
        "seed": 123,
        "ref_image_size": "match",
        "timeout": 60.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_api_workflow_references_existing_nodes_and_has_one_output() -> None:
    workflow = json.loads(
        (EXPERIMENT_DIR / "workflow_api.json").read_text(encoding="utf-8")
    )

    assert workflow["5"]["class_type"] == MODULE.NODE_TYPE
    assert workflow["13"]["class_type"] == "SaveImage"
    assert workflow["13"]["inputs"]["images"] == ["12", 0]
    for node in workflow.values():
        for value in node.get("inputs", {}).values():
            if (
                isinstance(value, list)
                and len(value) == 2
                and isinstance(value[0], str)
                and isinstance(value[1], int)
            ):
                assert value[0] in workflow
                assert value[1] >= 0


def test_build_workflow_applies_probe_arguments_without_mutating_template() -> None:
    template = json.loads(
        (EXPERIMENT_DIR / "workflow_api.json").read_text(encoding="utf-8")
    )

    workflow = MODULE.build_workflow(_args())

    assert workflow["4"]["inputs"]["image"] == "reference.png"
    assert workflow["5"]["inputs"]["width"] == 1024
    assert workflow["5"]["inputs"]["height"] == 768
    assert workflow["5"]["inputs"]["prompt"] == "Use <Picture 1> as the identity reference."
    assert workflow["8"]["inputs"]["noise_seed"] == 123
    assert workflow["10"]["inputs"]["steps"] == 12
    assert template["4"]["inputs"]["image"] == "comfy-installer-e2e-face.png"


def test_build_workflow_adds_contiguous_multi_reference_loaders() -> None:
    workflow = MODULE.build_workflow(
        _args(references=["character.png", "pose.png", "outfit.png"])
    )

    assert workflow["4"]["inputs"]["image"] == "character.png"
    assert workflow["14"]["inputs"]["image"] == "pose.png"
    assert workflow["15"]["inputs"]["image"] == "outfit.png"
    assert workflow["5"]["inputs"]["ref_image_2"] == ["14", 0]
    assert workflow["5"]["inputs"]["ref_image_3"] == ["15", 0]


def test_validate_inputs_rejects_unaligned_resolution_in_temp_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_dir = tmp_path / "comfy" / "input"
    input_dir.mkdir(parents=True)
    (input_dir / "reference.png").write_bytes(b"test")
    model_paths = tuple(tmp_path / f"model-{index}.safetensors" for index in range(3))
    for path in model_paths:
        path.write_bytes(b"test")
    monkeypatch.setattr(MODULE, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(MODULE, "REQUIRED_MODELS", model_paths)

    with pytest.raises(ValueError, match="32의 양의 배수"):
        MODULE.validate_inputs(_args(width=1000))


def test_sample_vram_records_first_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        MODULE,
        "request_json",
        lambda _url: {
            "devices": [{"vram_total": 16 * 1024**3, "vram_free": 6 * 1024**3}]
        },
    )
    observations = []

    MODULE.sample_vram("http://127.0.0.1:8188", observations)

    assert observations == [(16 * 1024**3, 6 * 1024**3)]
