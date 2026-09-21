from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from comfy_installer import production_image_diagnostic as diagnostic
from modes.asset_mode import AssetMode
from queue_manager import QueueItem, QueueManager


def _workflow() -> dict:
    return {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": "anima.safetensors"},
        },
        "2": {
            "class_type": "Power Lora Loader (rgthree)",
            "inputs": {
                "model": ["1", 0],
                "lora_1": {
                    "on": True,
                    "lora": "fixed.safetensors",
                    "strength": 0.5,
                },
            },
        },
        "3": {
            "class_type": "DCWModelPatch",
            "inputs": {
                "model": ["2", 0],
                "dcw_enabled": True,
                "cwm_enabled": True,
                "smc_preset": "Off",
            },
        },
        "4": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["3", 0],
                "positive": ["10", 0],
                "negative": ["11", 0],
                "latent_image": ["12", 0],
            },
        },
        "5": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["4", 0], "vae": ["13", 0]},
        },
        "6": {
            "class_type": "PreviewImage",
            "inputs": {"images": ["5", 0]},
        },
        "10": {"class_type": "Positive", "inputs": {}},
        "11": {"class_type": "Negative", "inputs": {}},
        "12": {"class_type": "EmptyLatent", "inputs": {}},
        "13": {"class_type": "VAELoader", "inputs": {}},
    }


def _node_by_class(workflow: dict, class_type: str) -> tuple[str, dict]:
    matches = [
        (str(node_id), node)
        for node_id, node in workflow.items()
        if node.get("class_type") == class_type
    ]
    assert len(matches) == 1
    return matches[0]


def test_abcd_graphs_preserve_production_path_and_insert_only_selected_boundary() -> None:
    cases = {
        "A": (True, False, False),
        "B": (True, True, False),
        "C": (False, False, False),
        "D": (True, False, True),
    }
    for name, (dcw_enabled, refresh, stable) in cases.items():
        workflow = copy.deepcopy(_workflow())
        AssetMode._apply_diagnostic_model_patch_override(workflow, dcw_enabled)
        stable_nodes = AssetMode._apply_diagnostic_stable_model_reuse(
            workflow,
            stable,
            f"session-{name}",
            f"run-{name}",
        )
        refresh_nodes = AssetMode._apply_diagnostic_model_patcher_refresh(
            workflow,
            refresh,
        )
        probes = AssetMode._apply_diagnostic_runtime_probes(workflow, f"run-{name}")

        sampler = workflow["4"]
        model_probe_id = str(sampler["inputs"]["model"][0])
        assert workflow[model_probe_id]["class_type"] == "SoyaDiagnosticModelProbe_mdsoya"
        assert workflow["5"]["inputs"]["samples"] != ["4", 0]
        preview_source = str(workflow["6"]["inputs"]["images"][0])
        assert workflow[preview_source]["class_type"] == "SoyaDiagnosticImageProbe_mdsoya"
        assert len(probes) == 2

        if name == "B":
            assert len(refresh_nodes) == 1
            refresh_id = refresh_nodes[0]["refresh_node_id"]
            assert workflow[model_probe_id]["inputs"]["model"] == [refresh_id, 0]
        else:
            assert refresh_nodes == []

        if name == "C":
            assert workflow["3"]["inputs"]["dcw_enabled"] is False
            assert workflow["3"]["inputs"]["cwm_enabled"] is False
        else:
            assert workflow["3"]["inputs"]["dcw_enabled"] is True
            assert workflow["3"]["inputs"]["cwm_enabled"] is True

        if name == "D":
            assert len(stable_nodes) == 1
            stable_id = stable_nodes[0]["reuse_node_id"]
            assert workflow["3"]["inputs"]["model"] == [stable_id, 0]
            assert workflow[stable_id]["inputs"]["model"] == ["2", 0]
        else:
            assert stable_nodes == []


def test_probe_parser_identifies_first_nonfinite_stage_and_complete_coverage() -> None:
    events = [
        {
            "event": "model_state",
            "stage": "model_before_sampler",
            "model": {"patches_uuid": "one"},
        },
        *[
            {
                "event": "tensor_state",
                "stage": stage,
                "model": (
                    {
                        "patches_uuid": "one",
                        "current_weight_patches_uuid": "one",
                    }
                    if stage == "latent_after_sampler"
                    else None
                ),
                "stats": {
                    "all_finite": stage != "latent_after_sampler",
                    "total_nonfinite": 4 if stage == "latent_after_sampler" else 0,
                    "tensor_count": 1,
                },
            }
            for stage in (
                "conditioning_positive",
                "conditioning_negative",
                "latent_before_sampler",
                "latent_after_sampler",
                "vae_output:5",
            )
        ],
    ]
    text = "\n".join(
        diagnostic._PROBE_LOG_PREFIX + json.dumps(event)
        for event in events
    )

    parsed = diagnostic._probe_events(text)
    summary = diagnostic._probe_summary(parsed)

    assert parsed == events
    assert summary["missing_stages"] == []
    assert summary["first_nonfinite_stage"] == "latent_after_sampler"
    assert summary["model_before_sampler"]["patches_uuid"] == "one"
    assert summary["model_after_sampler"]["current_weight_patches_uuid"] == "one"


def test_model_reuse_summary_exposes_weight_drift_and_uuid_transition_failure() -> None:
    def model(patcher: int, patch_uuid: str, current_uuid: str, weight_hash: str) -> dict:
        return {
            "patcher_id": patcher,
            "model_id": 9,
            "patches_uuid": patch_uuid,
            "current_weight_patches_uuid": current_uuid,
            "patch_structure_sha256": "fixed-structure",
            "patched_weight_samples": {"weights_sha256": weight_hash},
        }

    case = {
        "runs": [
            {
                "index": 0,
                "phase": "warmup",
                "probe_summary": {
                    "model_before_sampler": model(1, "warm", "base", "base-weight"),
                    "model_after_sampler": model(1, "warm", "warm", "good-weight"),
                    "stable_model_reuse": {
                        "cache_hit": False,
                        "structure_matches": True,
                        "incoming": {"patcher_id": 101},
                        "chosen": {"patcher_id": 101, "patches_uuid": "stable"},
                    },
                },
            },
            {
                "index": 1,
                "phase": "measurement",
                "probe_summary": {
                    "model_before_sampler": model(2, "one", "warm", "good-weight"),
                    "model_after_sampler": model(2, "one", "one", "good-weight"),
                    "stable_model_reuse": {
                        "cache_hit": True,
                        "structure_matches": True,
                        "incoming": {"patcher_id": 102},
                        "chosen": {"patcher_id": 101, "patches_uuid": "stable"},
                    },
                },
            },
            {
                "index": 2,
                "phase": "measurement",
                "probe_summary": {
                    "model_before_sampler": model(3, "two", "wrong", "good-weight"),
                    "model_after_sampler": model(3, "two", "two", "drifted-weight"),
                    "stable_model_reuse": {
                        "cache_hit": True,
                        "structure_matches": True,
                        "incoming": {"patcher_id": 103},
                        "chosen": {"patcher_id": 101, "patches_uuid": "stable"},
                    },
                },
            },
        ]
    }

    state = diagnostic._model_reuse_summary(case)

    assert state["resident_model_ids"] == [9]
    assert len(state["before_patcher_ids"]) == 2
    assert state["patch_structure_hashes"] == ["fixed-structure"]
    assert state["after_weight_state_hashes"] == ["good-weight", "drifted-weight"]
    assert state["previous_patch_uuid_transition_matches"] == 1
    assert len(state["previous_patch_uuid_transition_failures"]) == 1
    assert state["post_sampler_uuid_alignment_failures"] == []
    assert state["stable_cache_hits"] == 2
    assert state["stable_incoming_patcher_ids"] == [101, 102, 103]
    assert state["stable_chosen_patcher_ids"] == [101]


def test_blocky_color_noise_from_remote_failure_is_detected(tmp_path: Path) -> None:
    rng = np.random.default_rng(1234)
    gray = rng.integers(0, 256, size=(128, 88, 1), dtype=np.uint8)
    small = np.repeat(gray, 3, axis=2)
    blocky = np.repeat(np.repeat(small, 8, axis=0), 8, axis=1)
    path = tmp_path / "blocky-noise.png"
    Image.fromarray(blocky, mode="RGB").save(path)

    metrics = diagnostic._image_metrics(path)

    assert metrics["entropy"] >= 7.5
    assert metrics["neighbor_difference"] >= 10.0
    assert metrics["pixel_abnormal"] is True
    assert "블록형/고주파 컬러 노이즈 의심" in metrics["pixel_abnormal_reasons"]


def test_workflow_artifact_resolution_cannot_escape_models_folder(tmp_path: Path) -> None:
    comfy_root = tmp_path / "comfy"
    (comfy_root / "models" / "unet").mkdir(parents=True)
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"not a model from the managed models folder")

    matches = diagnostic._resolve_workflow_artifact(
        comfy_root,
        kind="unet",
        name="../../../outside.safetensors",
    )

    assert matches == []


def test_conclusions_treat_stable_reuse_only_recovery_as_actionable() -> None:
    cases = [
        {
            "name": name,
            "dcw_cwm_smc_enabled": name != "production_patch_off",
            "model_patcher_refresh": name == "production_model_refresh",
            "stable_model_reuse": name == "production_stable_model_reuse",
            "runs": [
                {
                    "index": 4,
                    "abnormal": name != "production_stable_model_reuse",
                    "abnormal_reasons": ["black"],
                    "probe_summary": {},
                }
            ],
        }
        for name in (
            "production_patch_on",
            "production_model_refresh",
            "production_patch_off",
            "production_stable_model_reuse",
        )
    ]

    conclusions = diagnostic._conclusions(cases)

    assert any("D만 ModelPatcher 반복 재구성을 제거해 안정화" in line for line in conclusions)


def test_transient_memory_profiles_isolate_one_runtime_variable() -> None:
    base = {
        "auto_start": True,
        "enable_cors": True,
        "listen_all": True,
        "fast": False,
        "disable_dynamic_vram": False,
        "vram_mode": "highvram",
        "cuda_device": None,
        "extra_args": "--preview-method none --async-offload 3 --disable-smart-memory",
    }

    dynamic_on = diagnostic._profile_with_memory_intervention(
        base,
        dynamic_vram=True,
    )
    assert dynamic_on["vram_mode"] == "auto"
    assert dynamic_on["disable_dynamic_vram"] is False
    assert "--async-offload 3" in dynamic_on["extra_args"]
    assert "--disable-smart-memory" in dynamic_on["extra_args"]

    async_off = diagnostic._profile_with_memory_intervention(
        base,
        disable_async_offload=True,
    )
    assert async_off["vram_mode"] == "highvram"
    assert "--async-offload" not in async_off["extra_args"]
    assert "--disable-async-offload" in async_off["extra_args"]
    assert "--disable-smart-memory" in async_off["extra_args"]
    assert "--preview-method none" in async_off["extra_args"]
    assert base["extra_args"].endswith("--disable-smart-memory")


def test_server_case_contract_contains_every_adaptive_memory_profile() -> None:
    assert diagnostic.PRODUCTION_IMAGE_DIAGNOSTIC_CASE_NAMES == {
        "production_patch_on",
        "production_model_refresh",
        "production_patch_off",
        "production_stable_model_reuse",
        "runtime_dynamic_on",
        "runtime_dynamic_off",
        "runtime_async_offload_off",
        "runtime_smart_memory_off",
        "runtime_pinned_memory_off",
    }


def test_adaptive_memory_plans_use_the_reproducing_workflow_case() -> None:
    base = {
        "disable_dynamic_vram": False,
        "vram_mode": "highvram",
        "extra_args": "--disable-smart-memory",
    }
    source = {
        "name": "production_model_refresh",
        "dcw_cwm_smc_enabled": True,
        "model_patcher_refresh": True,
        "stable_model_reuse": False,
    }

    plans, skipped = diagnostic._adaptive_memory_plans(base, source)

    assert all(plan["source_case"] == "production_model_refresh" for plan in plans)
    assert all(plan["model_patcher_refresh"] is True for plan in plans)
    assert "runtime_smart_memory_off" not in {plan["name"] for plan in plans}
    assert skipped == [
        {
            "name": "runtime_smart_memory_off",
            "reason": "원래 실행 프로필에 이미 같은 메모리 설정이 적용되어 별도 실행을 생략했습니다.",
        }
    ]


def test_restart_applies_transient_profile_without_mutating_restore_token() -> None:
    original_profile = {
        "disable_dynamic_vram": False,
        "vram_mode": "highvram",
        "extra_args": "",
    }
    transient_profile = {
        "disable_dynamic_vram": False,
        "vram_mode": "auto",
        "extra_args": "--disable-async-offload",
    }
    paused = {
        "instances": {
            "1": {
                "instance_id": 1,
                "port": 8188,
                "profile": copy.deepcopy(original_profile),
            }
        }
    }
    resumed_tokens = []

    def production_call(request: dict) -> dict:
        return {
            "queue_idle": True,
            "runtime_running": True,
            "runtime_ready": True,
            "instance_id": 1,
            "runtime": {
                "profile": copy.deepcopy(transient_profile),
                "command": ["python", "main.py", "--disable-async-offload"],
            },
            "comfy_log": "ready" if request.get("include_logs") else "",
        }

    def resume(token: dict) -> dict:
        resumed_tokens.append(copy.deepcopy(token))
        return {"status": "resumed"}

    result = diagnostic._restart_managed_runtime(
        production_call=production_call,
        pause_managed_comfy=lambda: copy.deepcopy(paused),
        resume_managed_comfy=resume,
        cancel_event=Event(),
        log=None,
        case_name="runtime_async_offload_off",
        runtime_profile=transient_profile,
    )

    assert result["pause"]["instances"]["1"]["profile"] == original_profile
    assert resumed_tokens[0]["instances"]["1"]["profile"] == transient_profile
    assert paused["instances"]["1"]["profile"] == original_profile


def test_conclusions_identify_async_offload_as_independent_recovery() -> None:
    cases = []
    for name in (
        "production_patch_on",
        "production_model_refresh",
        "production_patch_off",
        "production_stable_model_reuse",
        "runtime_dynamic_on",
        "runtime_dynamic_off",
        "runtime_async_offload_off",
        "runtime_smart_memory_off",
        "runtime_pinned_memory_off",
    ):
        abnormal = name not in {
            "production_stable_model_reuse",
            "runtime_async_offload_off",
        }
        cases.append(
            {
                "name": name,
                "dcw_cwm_smc_enabled": name != "production_patch_off",
                "model_patcher_refresh": name == "production_model_refresh",
                "stable_model_reuse": name == "production_stable_model_reuse",
                "source_case": (
                    "production_patch_on" if name.startswith("runtime_") else None
                ),
                "runs": [
                    {
                        "index": 7,
                        "phase": "measurement",
                        "abnormal": abnormal,
                        "abnormal_reasons": ["non-finite"] if abnormal else [],
                        "probe_summary": {},
                    }
                ],
            }
        )

    conclusions = diagnostic._conclusions(cases)

    assert any(
        "이상을 회피한 단독 런타임 개입: async offload OFF" in line
        for line in conclusions
    )
    assert any(
        "DynamicVRAM 하나만으로는 설명되지 않습니다" in line
        for line in conclusions
    )


def test_stable_reuse_returns_first_equivalent_patcher() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "comfy"
        / "custom_nodes"
        / "comfyui-soya-custom-nodes"
        / "soya_image_diagnostic.py"
    )
    spec = importlib.util.spec_from_file_location("test_soya_image_diagnostic", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._STABLE_MODELS.clear()

    base = SimpleNamespace(
        current_weight_patches_uuid=None,
        model_lowvram=False,
        lowvram_patch_counter=0,
        model_loaded_weight_memory=0,
        device="cpu",
    )
    first = SimpleNamespace(
        model=base,
        clone_base_uuid="base",
        patches_uuid="first",
        patches={},
        backup={},
        object_patches={},
        weight_wrapper_patches={},
        model_options={},
        weight_inplace_update=False,
    )
    second = SimpleNamespace(**{**first.__dict__, "patches_uuid": "second"})
    node = module.SoyaDiagnosticStableModelReuse_mdsoya()

    first_result = node.reuse(first, "session", "run-1")[0]
    second_result = node.reuse(second, "session", "run-2")[0]

    assert first_result is first
    assert second_result is first


@pytest.mark.asyncio
async def test_queue_forwards_all_diagnostic_controls_to_asset_mode() -> None:
    captured = {}

    class FakeAssetMode:
        async def generate(self, **kwargs):
            captured.update(kwargs)
            return {"success": True, "filename": "result.webp"}

    manager = QueueManager()
    manager.asset_mode = FakeAssetMode()
    manager.get_config = lambda: {"comfy_input_dir": ""}
    body = {
        "character": "diagnostic",
        "diagnostic_dcw_cwm_smc_enabled": True,
        "diagnostic_model_patcher_refresh": False,
        "diagnostic_stable_model_reuse": True,
        "diagnostic_session_key": "session-D",
        "diagnostic_run_key": "run-D-04",
        "diagnostic_capture_workflow": True,
    }
    item = QueueItem(
        id="diagnostic-forwarding",
        type="asset_generation",
        label="diagnostic",
        params={"body": body},
    )

    result = await manager._handle_asset_generation(item)

    assert result["success"] is True
    assert captured["diagnostic_dcw_cwm_smc_enabled"] is True
    assert captured["diagnostic_model_patcher_refresh"] is False
    assert captured["diagnostic_stable_model_reuse"] is True
    assert captured["diagnostic_session_key"] == "session-D"
    assert captured["diagnostic_run_key"] == "run-D-04"
    assert captured["diagnostic_capture_workflow"] is True
