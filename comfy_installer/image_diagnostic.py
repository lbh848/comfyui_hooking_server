from __future__ import annotations

import copy
import datetime
import json
import math
import os
import platform
import re
import shutil
import subprocess
import time
import traceback
import uuid
import zipfile
from pathlib import Path
from threading import Event
from typing import Any, Callable, Mapping, Sequence

import httpx

from comfy_runtime import (
    comfy_launch_profile_extra_args,
    normalize_comfy_launch_profiles,
    parse_comfy_extra_args,
)

from .e2e import (
    ComfyE2ECancelled,
    ComfyE2EError,
    ComfyProcess,
    WorkflowValidation,
    execute_prompt,
)
from .operations import isolated_subprocess_env, uv_python_path
from .runtime_state import git_head
from .workflow_library import resolve_distribution_selection


LogCallback = Callable[[str, str], None]
ComfyLogCallback = Callable[[str], None]
ProgressCallback = Callable[[dict[str, Any]], None]

BASELINE_RUNS = 24
COMPARISON_RUNS = 18
_MARKER = "comfy-installer-image-diagnostic-v1"
_NODE_DIRNAME = "comfy-installer-image-diagnostic"
_DIAGNOSTIC_CLASSES = {
    "model": "LBDiagnosticModelProbe",
    "trigger": "LBDiagnosticLatentTrigger",
    "latent": "LBDiagnosticLatentProbe",
    "image": "LBDiagnosticImageProbe",
}
_SAMPLER_CLASSES = {"KSampler", "SoyaFirstSampler_mdsoya"}
_VAE_DECODE_CLASSES = {"VAEDecode", "VAEDecodeTiled"}
_MODEL_PATCH_CLASSES = {
    "PathchSageAttentionKJ",
    "DCWModelPatch",
    "Power Lora Loader (rgthree)",
    "SoyaBatchLoraLoader_mdsoya",
}
_SAFE_SECTION_VALUES = {
    "FACE_ID_ACTIVATE": "false",
    "LORA_ACTIVATE": "false",
    "FACE_LORA_ACTIVATE": "false",
    "STYLE_ACTIVATE": "false",
    "STYLE_LORA_ACTIVATE": "false",
    "POSE_ACTIVATE": "false",
    "HRF_ACTIVATE": "false",
    "ANIMA_HRF_ACTIVATE": "false",
    "HRF_CONTROL_NET": "false",
    "FD_ACTIVATE": "false",
    "HD_ACTIVATE": "false",
    "ED_ACTIVATE": "false",
    "ANIMA_FD_ACTIVATE": "false",
    "ANIMA_HD_ACTIVATE": "false",
    "ANIMA_ED_ACTIVATE": "false",
    "LORA_DATA": '{"list":[]}',
    "FACE_LORA_DATA": '{"list":[]}',
    "STYLE_LORA_DATA": '{"list":[]}',
    "CACHE_PATH": '{"list":[]}',
    "N_IMG": "1",
}
_SECRET_ARG_TOKENS = ("key", "token", "secret", "password", "authorization")


class ImageDiagnosticError(RuntimeError):
    """이미지 반복 생성 진단을 안전하게 수행할 수 없음."""


def _log(log: LogCallback | None, message: str, level: str = "info") -> None:
    if log is not None:
        log(message, level)
    else:
        print(f"[COMFY_INSTALL][IMAGE_DIAGNOSTIC][{level.upper()}] {message}")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f"{path.name}.part")
    try:
        with part.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(part, path)
    except Exception as exc:
        print(
            "[COMFY_INSTALL][IMAGE_DIAGNOSTIC] 파일 기록 실패: "
            f"path={path}, error={type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        raise


def _write_json(path: Path, value: Any) -> None:
    _atomic_write(
        path,
        (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )


def _is_link(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], (str, int))
        and isinstance(value[1], int)
    )


def _replace_section(text: str, section: str, replacement: str) -> str:
    pattern = re.compile(
        rf"(?ms)(^\[{re.escape(section)}\]\r?\n).*?(?=^\[[A-Z0-9_]+\]\r?$|\Z)"
    )
    updated, _ = pattern.subn(
        lambda match: f"{match.group(1)}{replacement.rstrip()}\n",
        text,
    )
    return updated


def _replace_links(value: Any, replacements: Mapping[tuple[str, int], list]) -> Any:
    if _is_link(value):
        replacement = replacements.get((str(value[0]), int(value[1])))
        return copy.deepcopy(replacement) if replacement is not None else value
    if isinstance(value, dict):
        return {key: _replace_links(child, replacements) for key, child in value.items()}
    if isinstance(value, list):
        return [_replace_links(child, replacements) for child in value]
    return value


def _iter_links(value: Any):
    if _is_link(value):
        yield str(value[0]), int(value[1])
    elif isinstance(value, dict):
        for child in value.values():
            yield from _iter_links(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_links(child)


def _bypass_model_patches(prompt: dict) -> tuple[dict, list[dict[str, str]]]:
    result = copy.deepcopy(prompt)
    bypassed: list[dict[str, str]] = []
    bypassed_ids: set[str] = set()
    while True:
        replacements: dict[tuple[str, int], list] = {}
        removable: set[str] = set()
        for raw_node_id, node in result.items():
            if not isinstance(node, dict) or node.get("class_type") not in _MODEL_PATCH_CLASSES:
                continue
            inputs = node.get("inputs")
            if not isinstance(inputs, dict):
                continue
            node_id = str(raw_node_id)
            if node_id in bypassed_ids:
                continue
            output_sources: dict[int, list] = {}
            if _is_link(inputs.get("model")):
                output_sources[0] = copy.deepcopy(inputs["model"])
            if node.get("class_type") in {
                "Power Lora Loader (rgthree)", "SoyaBatchLoraLoader_mdsoya"
            } and _is_link(inputs.get("clip")):
                output_sources[1] = copy.deepcopy(inputs["clip"])
            used_outputs = {
                output_index
                for linked_id, output_index in _iter_links(result)
                if linked_id == node_id
            }
            if not used_outputs.issubset(output_sources):
                if node.get("class_type") == "SoyaBatchLoraLoader_mdsoya":
                    inputs["enable"] = "false"
                    inputs["lora_list"] = '{"list":[]}'
                    bypassed_ids.add(node_id)
                    bypassed.append(
                        {
                            "node_id": node_id,
                            "class_type": str(node.get("class_type")),
                            "mode": "disabled_in_place",
                        }
                    )
                    continue
                print(
                    "[COMFY_INSTALL][IMAGE_DIAGNOSTIC] patch 우회 생략: "
                    f"node={node_id}, class={node.get('class_type')}, "
                    f"대체 불가능 출력={sorted(used_outputs - set(output_sources))}"
                )
                continue
            for output_index, source in output_sources.items():
                replacements[(node_id, output_index)] = source
            removable.add(node_id)
            bypassed.append(
                {"node_id": node_id, "class_type": str(node.get("class_type"))}
            )
            bypassed_ids.add(node_id)
        if not replacements:
            break
        for _ in range(len(replacements)):
            resolved = {
                key: _replace_links(source, replacements)
                for key, source in replacements.items()
            }
            if resolved == replacements:
                break
            replacements = resolved
        result = _replace_links(result, replacements)
        for node_id in removable:
            result.pop(node_id, None)

    for node in result.values():
        if not isinstance(node, dict) or node.get("class_type") != "SoyaFirstSampler_mdsoya":
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        required = (
            "model", "seed", "steps", "cfg", "sampler_name", "scheduler",
            "positive", "negative", "latent_image", "denoise",
        )
        if not all(name in inputs for name in required):
            continue
        node["class_type"] = "KSampler"
        node["inputs"] = {name: copy.deepcopy(inputs[name]) for name in required}
        bypassed.append({"node_id": "sampler", "class_type": "SoyaFirstSampler_mdsoya"})
    return result, bypassed


def _find_sampler_for_decode(prompt: Mapping[str, Any], start_link: list) -> str | None:
    pending = [str(start_link[0])]
    visited: set[str] = set()
    while pending:
        node_id = pending.pop(0)
        if node_id in visited:
            continue
        visited.add(node_id)
        node = prompt.get(node_id)
        if not isinstance(node, dict):
            continue
        if node.get("class_type") in _SAMPLER_CLASSES:
            return node_id
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        for value in inputs.values():
            if _is_link(value):
                pending.append(str(value[0]))
    return None


def _ancestor_ids(prompt: Mapping[str, Any], start_ids: Sequence[str]) -> set[str]:
    pending = [str(value) for value in start_ids]
    visited: set[str] = set()
    while pending:
        node_id = pending.pop()
        if node_id in visited:
            continue
        visited.add(node_id)
        node = prompt.get(node_id)
        inputs = node.get("inputs") if isinstance(node, dict) else None
        if isinstance(inputs, dict):
            pending.extend(linked_id for linked_id, _ in _iter_links(inputs))
    return visited


def _sanitize_prompt(prompt: dict) -> dict:
    result = copy.deepcopy(prompt)
    for node in result.values():
        if not isinstance(node, dict) or not isinstance(node.get("inputs"), dict):
            continue
        inputs = node["inputs"]
        if isinstance(inputs.get("batch_size"), int):
            inputs["batch_size"] = 1
        if node.get("class_type") == "md_soya_InstantReferenceLoRA":
            inputs["preview_enable"] = False
        value = inputs.get("value")
        if node.get("class_type") != "PrimitiveStringMultiline" or not isinstance(value, str):
            continue
        for section, replacement in _SAFE_SECTION_VALUES.items():
            value = _replace_section(value, section, replacement)
        inputs["value"] = value
    return result


def prepare_diagnostic_prompt(
    validation: WorkflowValidation,
    *,
    output_dir: Path,
    profile_name: str,
    unpatched: bool = False,
) -> dict[str, Any]:
    prompt = _sanitize_prompt(validation.prompt)
    bypassed: list[dict[str, str]] = []
    if unpatched:
        prompt, bypassed = _bypass_model_patches(prompt)

    sampler_id = decode_id = None
    decode_link = None
    for raw_decode_id, node in prompt.items():
        if not isinstance(node, dict) or node.get("class_type") not in _VAE_DECODE_CLASSES:
            continue
        inputs = node.get("inputs")
        samples = inputs.get("samples") if isinstance(inputs, dict) else None
        if not _is_link(samples):
            continue
        candidate = _find_sampler_for_decode(prompt, samples)
        if candidate is not None:
            sampler_id = candidate
            decode_id = str(raw_decode_id)
            decode_link = copy.deepcopy(samples)
            break
    if sampler_id is None or decode_id is None or decode_link is None:
        classes = sorted(
            str(node.get("class_type")) for node in prompt.values() if isinstance(node, dict)
        )
        message = (
            "VAE Decode로 이어지는 주 sampler를 찾지 못했습니다. "
            f"workflow={validation.filename}, classes={classes}"
        )
        print(f"[COMFY_INSTALL][IMAGE_DIAGNOSTIC] {message}")
        raise ImageDiagnosticError(message)

    # The diagnostic executes only the first sampler -> VAE decode path from
    # the clean distribution workflow.  Keeping just the decode ancestors
    # removes every unrelated output branch without object_info schema
    # inspection and without requiring any face/training fixture.
    required_nodes = _ancestor_ids(prompt, [decode_id])
    prompt = {
        node_id: node
        for node_id, node in prompt.items()
        if node_id in required_nodes
    }

    sampler = prompt[sampler_id]
    sampler_inputs = sampler.get("inputs")
    if not isinstance(sampler_inputs, dict):
        raise ImageDiagnosticError(f"sampler 입력이 객체가 아닙니다: node={sampler_id}")
    model_link = sampler_inputs.get("model")
    latent_link = sampler_inputs.get("latent_image")
    if not _is_link(model_link) or not _is_link(latent_link):
        message = (
            "sampler 계측 삽입에 필요한 model/latent_image 연결이 없습니다: "
            f"node={sampler_id}, model={model_link!r}, latent={latent_link!r}"
        )
        print(f"[COMFY_INSTALL][IMAGE_DIAGNOSTIC] {message}")
        raise ImageDiagnosticError(message)

    ids = {
        "model": "__lb_diag_model_probe",
        "trigger": "__lb_diag_latent_trigger",
        "latent": "__lb_diag_latent_probe",
        "image": "__lb_diag_image_probe",
    }
    sampler_inputs["model"] = [ids["model"], 0]
    sampler_inputs["latent_image"] = [ids["trigger"], 0]
    prompt[decode_id]["inputs"]["samples"] = [ids["latent"], 0]
    prompt[ids["model"]] = {
        "class_type": _DIAGNOSTIC_CLASSES["model"],
        "inputs": {"model": model_link, "run_id": "pending", "trace_call": -1},
    }
    prompt[ids["trigger"]] = {
        "class_type": _DIAGNOSTIC_CLASSES["trigger"],
        "inputs": {"latent": latent_link, "nonce": "pending"},
    }
    prompt[ids["latent"]] = {
        "class_type": _DIAGNOSTIC_CLASSES["latent"],
        "inputs": {"latent": decode_link, "run_id": "pending"},
    }
    prompt[ids["image"]] = {
        "class_type": _DIAGNOSTIC_CLASSES["image"],
        "inputs": {
            "images": [decode_id, 0],
            "run_id": "pending",
            "profile": profile_name,
            "output_dir": str(output_dir.resolve()),
        },
    }
    sampler_mode = sampler_inputs.get("sampler_mode")
    return {
        "prompt": prompt,
        "ids": ids,
        "sampler": {
            "node_id": sampler_id,
            "class_type": str(sampler.get("class_type")),
            "sampler_mode": sampler_mode if isinstance(sampler_mode, str) else None,
            "steps": sampler_inputs.get("steps") if isinstance(sampler_inputs.get("steps"), int) else None,
            "decode_node_id": decode_id,
        },
        "bypassed": bypassed,
    }


def _prompt_for_run(template: dict, ids: Mapping[str, str], run_id: str, trace_call: int) -> dict:
    prompt = copy.deepcopy(template)
    prompt[ids["model"]]["inputs"]["run_id"] = run_id
    prompt[ids["model"]]["inputs"]["trace_call"] = int(trace_call)
    prompt[ids["trigger"]]["inputs"]["nonce"] = run_id
    prompt[ids["latent"]]["inputs"]["run_id"] = run_id
    prompt[ids["image"]]["inputs"]["run_id"] = run_id
    return prompt


def _parse_probe_output(execution: Mapping[str, Any], image_node_id: str) -> dict:
    output_data = execution.get("output_data")
    node_output = output_data.get(image_node_id) if isinstance(output_data, dict) else None
    values = node_output.get("diagnostic") if isinstance(node_output, dict) else None
    if not isinstance(values, list) or not values or not isinstance(values[0], str):
        message = (
            "이미지 계측 노드 결과가 history에 없습니다: "
            f"node={image_node_id}, output_keys={list(output_data or {})}"
        )
        print(f"[COMFY_INSTALL][IMAGE_DIAGNOSTIC] {message}")
        raise ImageDiagnosticError(message)
    try:
        payload = json.loads(values[0])
    except Exception as exc:
        print(
            "[COMFY_INSTALL][IMAGE_DIAGNOSTIC] 계측 JSON 해석 실패: "
            f"node={image_node_id}, value={values[0][:1000]!r}, error={exc}"
        )
        traceback.print_exc()
        raise ImageDiagnosticError(f"계측 JSON 해석 실패: {exc}") from exc
    if not isinstance(payload, dict):
        raise ImageDiagnosticError("계측 결과의 최상위 값이 객체가 아닙니다.")
    return payload


def _thumbnail_mae(left: Mapping[str, Any], right: Mapping[str, Any]) -> float | None:
    left_values = left.get("thumbnail_16x16_rgb")
    right_values = right.get("thumbnail_16x16_rgb")
    if not isinstance(left_values, list) or not isinstance(right_values, list):
        return None
    if not left_values or len(left_values) != len(right_values):
        return None
    try:
        return sum(abs(float(a) - float(b)) for a, b in zip(left_values, right_values)) / len(left_values)
    except Exception as exc:
        print(f"[COMFY_INSTALL][IMAGE_DIAGNOSTIC] thumbnail MAE 계산 실패: {exc}")
        traceback.print_exc()
        return None


def _stats_relative_mae(left: Mapping[str, Any], right: Mapping[str, Any]) -> float | None:
    left_values = left.get("sample_preview")
    right_values = right.get("sample_preview")
    if not isinstance(left_values, list) or not isinstance(right_values, list):
        return None
    if not left_values or len(left_values) != len(right_values):
        return None
    try:
        mae = sum(abs(float(a) - float(b)) for a, b in zip(left_values, right_values)) / len(left_values)
        scale = max(abs(float(right.get("std") or 0.0)), 0.01)
        return mae / scale
    except Exception as exc:
        print(f"[COMFY_INSTALL][IMAGE_DIAGNOSTIC] tensor sample MAE 계산 실패: {exc}")
        traceback.print_exc()
        return None


def _model_drift(run: Mapping[str, Any], reference: Mapping[str, Any]) -> tuple[float | None, int | None]:
    calls = run.get("model_calls") if isinstance(run.get("model_calls"), list) else []
    reference_calls = (
        reference.get("model_calls")
        if isinstance(reference.get("model_calls"), list)
        else []
    )
    maximum = None
    maximum_index = None
    for index, (call, reference_call) in enumerate(zip(calls, reference_calls)):
        if not isinstance(call, dict) or not isinstance(reference_call, dict):
            continue
        left = call.get("output")
        right = reference_call.get("output")
        if not isinstance(left, dict) or not isinstance(right, dict):
            continue
        value = _stats_relative_mae(left, right)
        if value is not None and (maximum is None or value > maximum):
            maximum = value
            maximum_index = index
    return maximum, maximum_index


def assess_run(run: dict, reference: dict | None) -> dict[str, Any]:
    reasons: list[str] = []
    stage = None
    model_calls = run.get("model_calls") if isinstance(run.get("model_calls"), list) else []
    model_bad = any(
        isinstance(call, dict)
        and isinstance(call.get("output"), dict)
        and not call["output"].get("finite", True)
        for call in model_calls
    )
    latent = run.get("latent") if isinstance(run.get("latent"), dict) else {}
    image = run.get("image") if isinstance(run.get("image"), dict) else {}
    if model_bad:
        stage = "diffusion_model"
        reasons.append("diffusion model 출력에 NaN/Inf 발생")
    if not latent.get("finite", True):
        stage = stage or "sampler_latent"
        reasons.append("sampler 최종 latent에 NaN/Inf 발생")
    if not image.get("finite", True):
        stage = stage or "vae_decode"
        reasons.append("VAE decode 이미지에 NaN/Inf 발생")
    luminance_std = image.get("luminance_std")
    black_fraction = image.get("black_fraction")
    edge_mean = image.get("edge_mean")
    if isinstance(black_fraction, (int, float)) and black_fraction >= 0.985:
        reasons.append("거의 전체가 검정인 이미지")
    if isinstance(luminance_std, (int, float)) and luminance_std <= 0.003:
        reasons.append("명암 변화가 거의 없는 평면 이미지")

    thumbnail_mae = None
    model_relative_mae = None
    model_drift_call = None
    latent_relative_mae = None
    if reference is not None:
        ref_image = reference.get("image") if isinstance(reference.get("image"), dict) else {}
        thumbnail_mae = _thumbnail_mae(image, ref_image)
        if thumbnail_mae is not None and thumbnail_mae >= 0.03:
            reasons.append(f"동일 seed 기준 영상 구조 변화(MAE={thumbnail_mae:.4f})")
        ref_edge = ref_image.get("edge_mean")
        if (
            isinstance(edge_mean, (int, float))
            and isinstance(ref_edge, (int, float))
            and edge_mean >= max(0.22, ref_edge * 2.5)
            and (thumbnail_mae or 0.0) >= 0.03
        ):
            reasons.append("기준 이미지 대비 고주파 컬러 노이즈 급증")
        model_relative_mae, model_drift_call = _model_drift(run, reference)
        if model_relative_mae is not None and model_relative_mae >= 0.05:
            reasons.append(
                "동일 seed인데 diffusion model 수치가 변화"
                f"(call={model_drift_call}, relative_MAE={model_relative_mae:.4f})"
            )
        reference_latent = reference.get("latent")
        if isinstance(reference_latent, dict) and latent:
            latent_relative_mae = _stats_relative_mae(latent, reference_latent)
            if latent_relative_mae is not None and latent_relative_mae >= 0.05:
                reasons.append(
                    f"동일 seed인데 sampler latent 수치가 변화(relative_MAE={latent_relative_mae:.4f})"
                )
    return {
        "abnormal": bool(reasons),
        "stage": stage,
        "reasons": reasons,
        "thumbnail_mae": thumbnail_mae,
        "model_relative_mae": model_relative_mae,
        "model_drift_call": model_drift_call,
        "latent_relative_mae": latent_relative_mae,
    }


def _trace_call_for(run: dict, reference: dict | None) -> int:
    calls = run.get("model_calls") if isinstance(run.get("model_calls"), list) else []
    reference_calls = (
        reference.get("model_calls")
        if isinstance(reference, dict) and isinstance(reference.get("model_calls"), list)
        else []
    )
    for index, call in enumerate(calls):
        output = call.get("output") if isinstance(call, dict) else None
        if isinstance(output, dict) and not output.get("finite", True):
            return index
        if index < len(reference_calls) and isinstance(output, dict):
            reference_output = reference_calls[index].get("output")
            if (
                isinstance(reference_output, dict)
                and (_stats_relative_mae(output, reference_output) or 0.0) >= 0.05
            ):
                return index
    for index, call in enumerate(calls):
        output = call.get("output") if isinstance(call, dict) else None
        if index < len(reference_calls) and isinstance(output, dict):
            reference_output = reference_calls[index].get("output")
            if (
                isinstance(reference_output, dict)
                and output.get("sample_sha256") != reference_output.get("sample_sha256")
            ):
                return index
    return 0


def _gpu_snapshot() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,driver_version,memory.used,memory.total,temperature.gpu,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
            check=False,
        )
        if completed.returncode != 0:
            print(
                "[COMFY_INSTALL][IMAGE_DIAGNOSTIC] nvidia-smi 계측 실패: "
                f"returncode={completed.returncode}, stderr={completed.stderr[:1000]}"
            )
            return {"available": False, "error": completed.stderr.strip()[:1000]}
        rows = []
        for line in completed.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            rows.append(
                {
                    "index": parts[0] if len(parts) > 0 else None,
                    "name": parts[1] if len(parts) > 1 else None,
                    "driver": parts[2] if len(parts) > 2 else None,
                    "memory_used_mib": parts[3] if len(parts) > 3 else None,
                    "memory_total_mib": parts[4] if len(parts) > 4 else None,
                    "temperature_c": parts[5] if len(parts) > 5 else None,
                    "utilization_percent": parts[6] if len(parts) > 6 else None,
                }
            )
        return {"available": True, "gpus": rows}
    except Exception as exc:
        print(f"[COMFY_INSTALL][IMAGE_DIAGNOSTIC] nvidia-smi 실행 실패: {exc}")
        traceback.print_exc()
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def _runtime_environment(python: Path, comfy_root: Path) -> dict[str, Any]:
    script = (
        "import importlib.metadata as m,json,os,platform,torch;"
        "names=['torch','torchvision','xformers','sageattention','numpy','pillow'];"
        "versions={};"
        "\nfor n in names:\n"
        "  try: versions[n]=m.version(n)\n"
        "  except Exception: versions[n]=None\n"
        "devices=[]\n"
        "for i in range(torch.cuda.device_count()):\n"
        "  p=torch.cuda.get_device_properties(i); devices.append({'index':i,'name':p.name,'total_memory':p.total_memory,'compute_capability':f'{p.major}.{p.minor}'})\n"
        "print(json.dumps({'python':platform.python_version(),'torch':torch.__version__,"
        "'torch_cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),"
        "'cuda_available':torch.cuda.is_available(),'cuda_devices':devices,'packages':versions,"
        "'allocator_env':{k:os.environ.get(k) for k in ['PYTORCH_CUDA_ALLOC_CONF','CUDA_LAUNCH_BLOCKING','TORCH_CUDNN_V8_API_ENABLED']}}))"
    )
    result: dict[str, Any] = {
        "platform": platform.platform(),
        "comfy_ref": None,
        "runtime": None,
        "gpu": _gpu_snapshot(),
    }
    try:
        result["comfy_ref"] = git_head(comfy_root)
    except Exception as exc:
        print(f"[COMFY_INSTALL][IMAGE_DIAGNOSTIC] Comfy Git ref 확인 실패: {exc}")
        traceback.print_exc()
        result["comfy_ref_error"] = f"{type(exc).__name__}: {exc}"
    custom_nodes = []
    custom_nodes_root = comfy_root / "custom_nodes"
    if custom_nodes_root.is_dir():
        for child in sorted(custom_nodes_root.iterdir(), key=lambda value: value.name.casefold()):
            if not child.is_dir() or child.name == _NODE_DIRNAME:
                continue
            item: dict[str, Any] = {"name": child.name, "git_ref": None}
            if (child / ".git").exists():
                try:
                    item["git_ref"] = git_head(child)
                except Exception as exc:
                    print(
                        "[COMFY_INSTALL][IMAGE_DIAGNOSTIC] custom node Git ref 확인 실패: "
                        f"node={child.name}, error={exc}"
                    )
                    traceback.print_exc()
                    item["error"] = f"{type(exc).__name__}: {exc}"
            custom_nodes.append(item)
    result["custom_nodes"] = custom_nodes
    try:
        completed = subprocess.run(
            [str(python), "-c", script],
            cwd=str(comfy_root),
            env=isolated_subprocess_env(),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            check=False,
        )
        if completed.returncode == 0:
            result["runtime"] = json.loads(completed.stdout.strip())
        else:
            print(
                "[COMFY_INSTALL][IMAGE_DIAGNOSTIC] Python 환경 계측 실패: "
                f"returncode={completed.returncode}, stderr={completed.stderr[:2000]}"
            )
            result["runtime_error"] = completed.stderr.strip()[:2000]
    except Exception as exc:
        print(f"[COMFY_INSTALL][IMAGE_DIAGNOSTIC] Python 환경 계측 실행 실패: {exc}")
        traceback.print_exc()
        result["runtime_error"] = f"{type(exc).__name__}: {exc}"
    return result


def _workflow_inventory(validation: WorkflowValidation) -> dict[str, Any]:
    model_inputs = {
        "ckpt_name", "unet_name", "diffusion_model", "vae_name",
        "clip_name", "clip_name1", "clip_name2", "lora_name",
    }
    models: list[dict[str, str]] = []
    for node in validation.prompt.values():
        if not isinstance(node, dict) or not isinstance(node.get("inputs"), dict):
            continue
        for input_name, value in node["inputs"].items():
            if input_name not in model_inputs or not isinstance(value, str) or not value.strip():
                continue
            models.append(
                {
                    "class_type": str(node.get("class_type")),
                    "input": input_name,
                    "filename": Path(value).name,
                }
            )
    return {
        "classes": list(validation.classes),
        "models": models,
    }


def _redact_args(arguments: Sequence[str]) -> list[str]:
    result: list[str] = []
    redact_next = False
    for raw in arguments:
        value = str(raw)
        folded = value.casefold()
        if redact_next:
            result.append("<redacted>")
            redact_next = False
            continue
        if any(token in folded for token in _SECRET_ARG_TOKENS):
            if "=" in value:
                result.append(value.split("=", 1)[0] + "=<redacted>")
            else:
                result.append(value)
                redact_next = True
            continue
        result.append(value)
    return result


def _install_probe_node(comfy_root: Path) -> Path:
    source = Path(__file__).resolve().parent / "resources" / "image_diagnostic_node"
    custom_nodes_root = (comfy_root / "custom_nodes").resolve()
    target = (custom_nodes_root / _NODE_DIRNAME).resolve()
    if target.parent != custom_nodes_root or target.name != _NODE_DIRNAME:
        raise ImageDiagnosticError(f"임시 진단 노드 대상 경로가 안전하지 않습니다: {target}")
    marker = target / ".comfy-installer-owned"
    if not source.is_dir():
        raise ImageDiagnosticError(f"진단 노드 리소스가 없습니다: {source}")
    if target.exists():
        owned = marker.is_file() and marker.read_text(encoding="utf-8").strip() == _MARKER
        if not owned:
            raise ImageDiagnosticError(
                f"동일 이름의 사용자 custom node가 있어 덮어쓰지 않습니다: {target.name}"
            )
        shutil.rmtree(target)
    try:
        shutil.copytree(source, target)
        return target
    except Exception as exc:
        print(
            "[COMFY_INSTALL][IMAGE_DIAGNOSTIC] 임시 진단 노드 설치 실패: "
            f"source={source}, target={target}, error={exc}"
        )
        traceback.print_exc()
        raise ImageDiagnosticError(f"임시 진단 노드 설치 실패: {exc}") from exc


def _remove_probe_node(target: Path | None, comfy_root: Path) -> bool:
    if target is None or not target.exists():
        return True
    try:
        custom_nodes_root = (comfy_root / "custom_nodes").resolve()
        resolved = target.resolve()
        if resolved.parent != custom_nodes_root or resolved.name != _NODE_DIRNAME:
            print(
                "[COMFY_INSTALL][IMAGE_DIAGNOSTIC] 임시 노드 정리 거부: "
                f"대상 경로 범위 불일치, target={resolved}, root={custom_nodes_root}"
            )
            return False
        marker = target / ".comfy-installer-owned"
        if not marker.is_file() or marker.read_text(encoding="utf-8").strip() != _MARKER:
            print(
                "[COMFY_INSTALL][IMAGE_DIAGNOSTIC] 임시 노드 정리 거부: "
                f"소유권 marker 불일치, target={target}"
            )
            return False
        shutil.rmtree(resolved)
        return True
    except Exception as exc:
        print(
            "[COMFY_INSTALL][IMAGE_DIAGNOSTIC] 임시 진단 노드 정리 실패: "
            f"target={target}, error={exc}"
        )
        traceback.print_exc()
        return False


def _copy_process_log(process: ComfyProcess, report_dir: Path, profile_name: str) -> str | None:
    source = process.output_log_path
    if not source.is_file():
        print(
            "[COMFY_INSTALL][IMAGE_DIAGNOSTIC] Comfy 원본 로그가 없습니다: "
            f"profile={profile_name}, path={source}"
        )
        return None
    target = report_dir / "logs" / f"{profile_name}.log"
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        return str(target.relative_to(report_dir)).replace("\\", "/")
    except Exception as exc:
        print(
            "[COMFY_INSTALL][IMAGE_DIAGNOSTIC] Comfy 로그 복사 실패: "
            f"source={source}, target={target}, error={exc}"
        )
        traceback.print_exc()
        return None


def _profile_with(
    base: Mapping[str, Any],
    *,
    vram_mode: str | None = None,
    dynamic_off: bool | None = None,
    cpu_vae: bool | None = None,
) -> dict[str, Any]:
    profile = copy.deepcopy(dict(base))
    if vram_mode is not None:
        profile["vram_mode"] = vram_mode
    if dynamic_off is not None:
        profile["disable_dynamic_vram"] = dynamic_off
    extra_values = list(parse_comfy_extra_args(str(profile.get("extra_args") or "")))
    if cpu_vae is not None:
        extra_values = [value for value in extra_values if value != "--cpu-vae"]
        if cpu_vae:
            extra_values.append("--cpu-vae")
    profile["extra_args"] = subprocess.list2cmdline(extra_values)
    return profile


def _workflow_binding(
    config: Mapping[str, Any],
    *,
    workflow_library_root: Path,
    workflow_release: str,
) -> tuple[str, Path, str]:
    workflow_type = str(config.get("asset_workflow_type") or "ilxl").strip().casefold()
    mapping = {
        "ilxl": "asset_workflow_source_path",
        "regular": "asset_workflow_source_path",
        "anima_ilxl": "anima_asset_workflow_source_path",
        "anima": "anima_asset_workflow_source_path",
        "anima_only": "anima_only_asset_workflow_source_path",
    }
    binding = mapping.get(workflow_type, "anima_only_asset_workflow_source_path")
    try:
        selection = resolve_distribution_selection(
            library_root=workflow_library_root,
            release_version=workflow_release,
            selected_item_ids=[binding],
        )
    except Exception as exc:
        print(
            "[COMFY_INSTALL][IMAGE_DIAGNOSTIC] 팩 배포 원본 선택 실패: "
            f"release={workflow_release}, binding={binding}, error={exc}"
        )
        traceback.print_exc()
        raise ImageDiagnosticError(
            "현재 에셋 타입의 깨끗한 팩 워크플로우를 찾지 못했습니다: "
            f"release={workflow_release}, binding={binding}"
        ) from exc
    raw_path = selection.workflow_bindings.get(binding)
    path = Path(str(raw_path or "")).resolve()
    if not raw_path or not path.is_file():
        message = (
            "팩에서 선택한 에셋 워크플로우 원본이 없습니다: "
            f"release={workflow_release}, binding={binding}, path={path}"
        )
        print(f"[COMFY_INSTALL][IMAGE_DIAGNOSTIC] {message}")
        raise ImageDiagnosticError(message)
    return binding, path, workflow_type


def _convert_workflow_once(
    *,
    base_url: str,
    workflow_path: Path,
    binding: str,
    cancel_event: Event,
) -> WorkflowValidation:
    if cancel_event.is_set():
        raise ComfyE2ECancelled("이미지 진단 워크플로우 변환 전 중단되었습니다.")
    try:
        workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
        if not isinstance(workflow, dict) or not isinstance(workflow.get("nodes"), list):
            raise ImageDiagnosticError(
                f"팩 워크플로우에 nodes 배열이 없습니다: {workflow_path.name}"
            )
        with httpx.Client(
            base_url=base_url,
            timeout=httpx.Timeout(180, connect=15),
        ) as client:
            response = client.post("/workflow/convert", json=workflow)
            if response.status_code != 200:
                raise ImageDiagnosticError(
                    "팩 워크플로우 API 변환 실패: "
                    f"{workflow_path.name}, status={response.status_code}, "
                    f"body={response.text[:2000]}"
                )
            prompt = response.json()
        if isinstance(prompt, dict) and isinstance(prompt.get("prompt"), dict):
            prompt = prompt["prompt"]
        if not isinstance(prompt, dict) or not prompt:
            raise ImageDiagnosticError(
                f"팩 워크플로우 변환 결과가 비어 있습니다: {workflow_path.name}"
            )
        classes = tuple(
            sorted(
                {
                    str(node.get("class_type"))
                    for node in prompt.values()
                    if isinstance(node, dict)
                    and isinstance(node.get("class_type"), str)
                    and node.get("class_type")
                }
            )
        )
        return WorkflowValidation(
            binding_keys=(binding,),
            filename=workflow_path.name,
            node_count=len(prompt),
            class_count=len(classes),
            classes=classes,
            prompt=prompt,
            workflow=workflow,
        )
    except (ComfyE2ECancelled, ImageDiagnosticError):
        raise
    except Exception as exc:
        print(
            "[COMFY_INSTALL][IMAGE_DIAGNOSTIC] 팩 워크플로우 1회 변환 실패: "
            f"path={workflow_path}, error={exc}"
        )
        traceback.print_exc()
        raise ImageDiagnosticError(
            f"팩 워크플로우 1회 변환 실패: {workflow_path.name}: {exc}"
        ) from exc


def _run_profile(
    *,
    comfy_root: Path,
    python: Path,
    validation: WorkflowValidation,
    report_dir: Path,
    profile_name: str,
    profile: Mapping[str, Any],
    run_limit: int,
    cancel_event: Event,
    log: LogCallback | None,
    comfy_log: ComfyLogCallback | None,
    progress: ProgressCallback | None,
    unpatched: bool = False,
) -> dict[str, Any]:
    arguments = comfy_launch_profile_extra_args(profile)
    profile_result: dict[str, Any] = {
        "name": profile_name,
        "arguments": _redact_args(arguments),
        "settings": {
            "vram_mode": profile.get("vram_mode"),
            "disable_dynamic_vram": bool(profile.get("disable_dynamic_vram")),
            "cpu_vae": "--cpu-vae" in arguments,
        },
        "run_limit": run_limit,
        "unpatched": unpatched,
        "runs": [],
        "error": None,
        "log": None,
    }
    process = ComfyProcess(
        comfy_root=comfy_root,
        python=python,
        cancel_event=cancel_event,
        log=comfy_log,
        extra_args=arguments,
        # Image corruption diagnosis only needs a runnable Comfy API and the
        # target workflow.  Manager is not involved in sampling, so its API
        # version must never prevent evidence collection on the user's
        # already-installed environment.
        verify_manager=False,
        enable_manager=False,
    )
    try:
        _log(log, f"[이미지 진단] {profile_name} 독립 Comfy 기동: {' '.join(_redact_args(arguments)) or '기본 옵션'}")
        stats = process.start(timeout=900)
        profile_result["comfy_system"] = stats.get("system", {}) if isinstance(stats, dict) else {}
        prepared = prepare_diagnostic_prompt(
            validation,
            output_dir=report_dir / "images",
            profile_name=profile_name,
            unpatched=unpatched,
        )
        profile_result["sampler"] = prepared["sampler"]
        profile_result["bypassed"] = prepared["bypassed"]
        reference = None
        abnormal_run = None
        trace_pending = False
        trace_call = -1
        index = 1
        while index <= run_limit or trace_pending:
            if cancel_event.is_set():
                raise ComfyE2ECancelled(f"이미지 깨짐 검사 중단: profile={profile_name}")
            is_trace = trace_pending
            trace_pending = False
            run_id = f"{index:03d}{'-trace' if is_trace else ''}-{uuid.uuid4().hex[:8]}"
            if progress:
                progress(
                    {
                        "event": "image_diagnostic_run",
                        "profile": profile_name,
                        "current": index,
                        "total": run_limit,
                        "trace": is_trace,
                    }
                )
            _log(
                log,
                f"[이미지 진단] {profile_name} {index}/{run_limit}회"
                + (f" · 정밀 추적 call={trace_call}" if is_trace else ""),
            )
            before_gpu = _gpu_snapshot()
            started = time.monotonic()
            execution = execute_prompt(
                base_url=process.base_url,
                prompt=_prompt_for_run(prepared["prompt"], prepared["ids"], run_id, trace_call if is_trace else -1),
                workflow=validation.workflow,
                filename=f"image-diagnostic-{profile_name}-{index}",
                cancel_event=cancel_event,
                log=(lambda message: _log(log, message, "info")),
                timeout=3600,
                fatal_error=process.fatal_error,
            )
            measured = _parse_probe_output(execution, prepared["ids"]["image"])
            measured["index"] = index
            measured["trace"] = is_trace
            measured["duration_seconds"] = round(time.monotonic() - started, 3)
            measured["gpu_before"] = before_gpu
            measured["gpu_after"] = _gpu_snapshot()
            assessment = assess_run(measured, reference)
            measured["assessment"] = assessment
            profile_result["runs"].append(measured)
            if reference is None:
                reference = measured
            if assessment["abnormal"] and abnormal_run is None:
                abnormal_run = index
                profile_result["first_abnormal_run"] = index
                profile_result["first_abnormal_reasons"] = assessment["reasons"]
                trace_call = _trace_call_for(measured, reference)
                if not is_trace:
                    trace_pending = True
                _log(
                    log,
                    f"[이미지 진단] {profile_name} {index}회에서 이상 감지: "
                    + "; ".join(assessment["reasons"]),
                    "warning",
                )
            if abnormal_run is not None and is_trace:
                break
            index += 1
        profile_result["completed_runs"] = len(profile_result["runs"])
        profile_result["stable_runs"] = (
            abnormal_run - 1 if abnormal_run is not None else len(profile_result["runs"])
        )
        return profile_result
    except ComfyE2ECancelled:
        raise
    except Exception as exc:
        print(
            "[COMFY_INSTALL][IMAGE_DIAGNOSTIC] 프로필 실행 실패: "
            f"profile={profile_name}, args={_redact_args(arguments)}, error={exc}"
        )
        traceback.print_exc()
        profile_result["error"] = f"{type(exc).__name__}: {exc}"
        profile_result["completed_runs"] = len(profile_result["runs"])
        return profile_result
    finally:
        process.stop()
        profile_result["log"] = _copy_process_log(process, report_dir, profile_name)


def _conclusions(profiles: Sequence[Mapping[str, Any]]) -> list[str]:
    by_name = {str(profile.get("name")): profile for profile in profiles}
    conclusions: list[str] = []
    abnormal = [profile for profile in profiles if profile.get("first_abnormal_run")]
    successful = [
        profile for profile in profiles
        if not profile.get("error") and int(profile.get("completed_runs") or 0) > 0
    ]
    if not abnormal:
        completed = sum(int(profile.get("completed_runs") or 0) for profile in profiles)
        if not successful:
            conclusions.append(
                "유효한 반복 생성 결과를 얻지 못했습니다. 프로필별 실행 오류와 Comfy 원본 로그를 확인해야 합니다."
            )
            return conclusions
        conclusions.append(
            f"총 {completed}회에서 검정·노이즈·NaN/Inf·동일 seed 대규모 변화가 재현되지 않았습니다. "
            "미재현은 하드웨어/런타임 정상 판정이 아니라 이번 제한 횟수 내 관찰 결과입니다."
        )
        if len(successful) != len(profiles):
            conclusions.append("일부 비교 프로필은 실행에 실패했으므로 전체 가설 비교는 완료되지 않았습니다.")
        return conclusions

    for profile in abnormal:
        name = str(profile.get("name"))
        run = profile.get("first_abnormal_run")
        reasons = "; ".join(str(v) for v in profile.get("first_abnormal_reasons", []))
        conclusions.append(f"{name}: {run}회에서 최초 이상 감지 — {reasons}")

    novram = next(
        (
            profile for profile in profiles
            if profile.get("settings", {}).get("vram_mode") == "novram"
            and profile.get("settings", {}).get("disable_dynamic_vram") is True
        ),
        None,
    )
    novram_dynamic_on = next(
        (
            profile for profile in profiles
            if profile.get("settings", {}).get("vram_mode") == "novram"
            and profile.get("settings", {}).get("disable_dynamic_vram") is False
        ),
        None,
    )
    cpu_vae = next(
        (
            profile for profile in profiles
            if profile.get("settings", {}).get("cpu_vae") is True
        ),
        None,
    )
    core = by_name.get("core_unpatched")
    non_novram_failed = any(
        profile.get("first_abnormal_run")
        for profile in profiles
        if profile.get("settings", {}).get("vram_mode") != "novram"
    )
    if non_novram_failed and novram and not novram.get("first_abnormal_run") and not novram.get("error"):
        conclusions.append(
            "현재 설정은 실패했지만 NO_VRAM + Dynamic OFF는 제한 횟수 동안 유지됐습니다. "
            "단순 VRAM 총량 부족보다 GPU 상주 모델/버퍼의 반복 재사용 상태와 강한 상관관계가 있습니다."
        )
        if novram_dynamic_on and not novram_dynamic_on.get("error"):
            if novram_dynamic_on.get("first_abnormal_run"):
                conclusions.append(
                    "NO_VRAM에서도 Dynamic VRAM ON일 때만 재현되어 NO_VRAM 자체보다 Dynamic VRAM 상호작용의 영향이 큽니다."
                )
            else:
                conclusions.append(
                    "NO_VRAM은 Dynamic VRAM ON/OFF 양쪽에서 제한 횟수를 유지해, 이번 관찰에서는 적극적인 unload/offload가 핵심 차이입니다."
                )
    if cpu_vae and cpu_vae.get("first_abnormal_run"):
        conclusions.append("CPU VAE에서도 재현되어 GPU VAE 단독 원인 가설은 지지되지 않습니다.")
    if core and not core.get("first_abnormal_run") and not core.get("error") and abnormal:
        conclusions.append(
            "SageAttention/DCW/LoRA patch 및 1ST SAMPLER 우회 경로에서는 제한 횟수 동안 유지됐습니다. "
            "우회된 patch 조합을 우선 조사해야 합니다."
        )
    stages: set[str] = set()
    for profile in abnormal:
        for run in profile.get("runs", []):
            assessment = run.get("assessment") if isinstance(run, dict) else None
            if isinstance(assessment, dict) and assessment.get("stage"):
                stages.add(str(assessment["stage"]))
    if "diffusion_model" in stages:
        conclusions.append("NaN/Inf가 diffusion model 출력에서 이미 관찰되어 VAE decode 이전에 손상이 시작됩니다.")
    elif "sampler_latent" in stages:
        conclusions.append("모델 호출 뒤 sampler 최종 latent에서 최초 NaN/Inf가 관찰됐습니다.")
    elif "vae_decode" in stages:
        conclusions.append("sampler latent는 유한하지만 VAE decode 이미지에서 최초 NaN/Inf가 관찰됐습니다.")
    samplers = {
        str(profile.get("sampler", {}).get("class_type"))
        for profile in successful
        if isinstance(profile.get("sampler"), dict)
    }
    if samplers == {"KSampler"}:
        conclusions.append(
            "실제 계측 경로는 기본 KSampler였습니다. 이 실행에서는 1ST SAMPLER의 regional LoRA 경로가 사용되지 않았습니다."
        )
    return conclusions


def _render_report(result: Mapping[str, Any]) -> str:
    environment = result.get("environment", {})
    workflow = result.get("workflow", {})
    profiles = result.get("profiles", [])
    lines = [
        "# ComfyUI 이미지 깨짐 자동 진단",
        "",
        f"- 검사 ID: `{result.get('diagnostic_id')}`",
        f"- 생성 시각: {result.get('completed_at')}",
        f"- 워크플로우: `{workflow.get('filename')}` ({workflow.get('binding')})",
        f"- 에셋 타입: `{workflow.get('asset_workflow_type')}`",
        f"- Comfy ref: `{environment.get('comfy_ref')}`",
        "",
        "## 자동 판정",
        "",
    ]
    for conclusion in result.get("conclusions", []):
        lines.append(f"- {conclusion}")
    if result.get("errors"):
        lines.extend(["", "## 진단 중 오류", ""])
        for error in result["errors"]:
            lines.append(f"- {error}")
    lines.extend(
        [
            "",
            "## 프로필별 반복 결과",
            "",
            "| 프로필 | 실행 인자 | 완료 | 최초 이상 | 결과 |",
            "|---|---|---:|---:|---|",
        ]
    )
    for profile in profiles:
        error = profile.get("error")
        first = profile.get("first_abnormal_run")
        status = f"실행 오류: {error}" if error else ("이상 감지" if first else "제한 횟수 내 미재현")
        args = " ".join(profile.get("arguments", [])) or "기본"
        lines.append(
            f"| `{profile.get('name')}` | `{args}` | {profile.get('completed_runs', 0)} | "
            f"{first or '-'} | {status} |"
        )
    lines.extend(["", "## 회차별 경계 계측", ""])
    for profile in profiles:
        lines.extend(
            [
                f"### {profile.get('name')}",
                "",
                f"- sampler: `{json.dumps(profile.get('sampler'), ensure_ascii=False)}`",
                f"- 우회 노드: `{json.dumps(profile.get('bypassed'), ensure_ascii=False)}`",
                f"- Comfy 로그: `{profile.get('log') or '없음'}`",
                "",
                "| 회차 | 모델 유한 | latent 유한 | 이미지 유한 | 밝기 평균/표준편차 | edge | 기준 MAE | 판정 | 이미지 |",
                "|---:|---|---|---|---|---:|---:|---|---|",
            ]
        )
        for run in profile.get("runs", []):
            calls = run.get("model_calls", [])
            model_finite = all(call.get("output", {}).get("finite", True) for call in calls if isinstance(call, dict))
            latent = run.get("latent", {})
            image = run.get("image", {})
            assessment = run.get("assessment", {})
            reasons = "; ".join(assessment.get("reasons", [])) or "정상 범위"
            mae = assessment.get("thumbnail_mae")
            lines.append(
                f"| {run.get('index')} {'(정밀)' if run.get('trace') else ''} | {model_finite} | "
                f"{latent.get('finite')} | {image.get('finite')} | "
                f"{image.get('luminance_mean', 0):.4f}/{image.get('luminance_std', 0):.4f} | "
                f"{image.get('edge_mean', 0):.4f} | {mae:.4f} | {reasons} | "
                f"`images/{run.get('artifact')}` |" if isinstance(mae, (int, float)) else
                f"| {run.get('index')} {'(정밀)' if run.get('trace') else ''} | {model_finite} | "
                f"{latent.get('finite')} | {image.get('finite')} | "
                f"{image.get('luminance_mean', 0):.4f}/{image.get('luminance_std', 0):.4f} | "
                f"{image.get('edge_mean', 0):.4f} | - | {reasons} | `images/{run.get('artifact')}` |"
            )
            transitions = run.get("deep_transitions", [])
            if transitions:
                lines.append("")
                lines.append("최초 유한 입력 → 비유한 출력 모듈 후보:")
                for transition in transitions[:20]:
                    lines.append(
                        f"- `{transition.get('module')}` ({transition.get('module_type')}): "
                        "입력 finite=True → 출력 finite=False"
                    )
                lines.append("")
    lines.extend(
        [
            "",
            "## 해석 주의사항",
            "",
            "- 동일 seed와 동일 워크플로우를 반복했으므로 큰 영상 변화는 프롬프트 의미 평가가 아니라 실행 상태 변화 신호입니다.",
            "- NO_VRAM에서 제한 횟수를 통과해도 영구 해결을 의미하지 않습니다. CPU VAE가 12장 이후 실패했던 사례를 고려해 짧은 성공을 해결로 판정하지 않습니다.",
            "- `environment.json`과 `runs.json`에는 원본 프롬프트/API 키 없이 런타임·수치 계측만 저장했습니다.",
            "",
        ]
    )
    return "\n".join(lines)


def _make_archive(report_dir: Path, archive_path: Path) -> None:
    part = archive_path.with_name(f"{archive_path.name}.part")
    try:
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(part, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in sorted(report_dir.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(report_dir).as_posix())
        os.replace(part, archive_path)
    except Exception as exc:
        print(
            "[COMFY_INSTALL][IMAGE_DIAGNOSTIC] ZIP 생성 실패: "
            f"source={report_dir}, archive={archive_path}, error={exc}"
        )
        traceback.print_exc()
        raise ImageDiagnosticError(f"진단 ZIP 생성 실패: {exc}") from exc


def _public_result(result: Mapping[str, Any], archive_path: Path) -> dict[str, Any]:
    return {
        "operation": "image_diagnostic",
        "diagnostic_id": result.get("diagnostic_id"),
        "completed_at": result.get("completed_at"),
        "duration_seconds": result.get("duration_seconds"),
        "workflow": copy.deepcopy(result.get("workflow", {})),
        "conclusions": list(result.get("conclusions", [])),
        "errors": list(result.get("errors", [])),
        "incomplete": bool(result.get("incomplete")),
        "archive_id": result.get("archive_id"),
        "archive_name": result.get("archive_name"),
        "archive_size": archive_path.stat().st_size,
        "profile_summary": [
            {
                "name": profile.get("name"),
                "settings": copy.deepcopy(profile.get("settings", {})),
                "completed_runs": profile.get("completed_runs", 0),
                "first_abnormal_run": profile.get("first_abnormal_run"),
                "error": profile.get("error"),
            }
            for profile in result.get("profiles", [])
            if isinstance(profile, Mapping)
        ],
    }


def _remove_expanded_report(report_dir: Path, root: Path) -> None:
    try:
        resolved = report_dir.resolve()
        if resolved.parent != root.resolve() or not resolved.name:
            print(
                "[COMFY_INSTALL][IMAGE_DIAGNOSTIC] 펼친 진단 폴더 정리 거부: "
                f"report_dir={resolved}, root={root.resolve()}"
            )
            return
        shutil.rmtree(resolved)
    except Exception as exc:
        print(
            "[COMFY_INSTALL][IMAGE_DIAGNOSTIC] ZIP 생성 후 임시 폴더 정리 실패: "
            f"report_dir={report_dir}, error={exc}"
        )
        traceback.print_exc()


def run_image_diagnostic(
    *,
    project_root: Path,
    comfy_root: Path,
    config_path: Path,
    workflow_library_root: Path,
    workflow_release: str,
    cancel_event: Event,
    log: LogCallback | None = None,
    comfy_log: ComfyLogCallback | None = None,
    progress: ProgressCallback | None = None,
    pause_managed_comfy: Callable[[], Any] | None = None,
    resume_managed_comfy: Callable[[Any], Any] | None = None,
) -> dict[str, Any]:
    diagnostic_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "-" + uuid.uuid4().hex[:8]
    root = project_root / ".work" / "comfy-installer" / "image-diagnostics"
    report_dir = root / diagnostic_id
    archive_path = root / f"{diagnostic_id}.zip"
    report_dir.mkdir(parents=True, exist_ok=False)
    python = uv_python_path(comfy_root / ".venv")

    errors: list[str] = []
    profiles_result: list[dict[str, Any]] = []
    environment: dict[str, Any] = {}
    workflow_summary: dict[str, Any] = {}
    probe_node: Path | None = None
    pause_token: Any = None
    paused = False
    started = time.monotonic()
    try:
        if not python.is_file():
            raise ImageDiagnosticError(f"내장 Comfy Python이 없습니다: {python}")
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if not isinstance(config, dict):
                raise ImageDiagnosticError("config.json 최상위 값이 객체가 아닙니다.")
        except ImageDiagnosticError:
            raise
        except Exception as exc:
            print(f"[COMFY_INSTALL][IMAGE_DIAGNOSTIC] config 읽기 실패: {exc}")
            traceback.print_exc()
            raise ImageDiagnosticError(f"config.json 읽기 실패: {exc}") from exc

        binding, workflow_path, workflow_type = _workflow_binding(
            config,
            workflow_library_root=workflow_library_root,
            workflow_release=workflow_release,
        )
        workflow_summary = {
            "binding": binding,
            "filename": workflow_path.name,
            "asset_workflow_type": workflow_type,
            "source": "pack_distribution_original",
            "release": workflow_release,
        }
        launch_profiles = normalize_comfy_launch_profiles(config.get("comfy_launch_profiles"))
        current_profile = launch_profiles["1"]
        environment = _runtime_environment(python, comfy_root)
        environment["current_profile"] = {
            "vram_mode": current_profile.get("vram_mode"),
            "disable_dynamic_vram": current_profile.get("disable_dynamic_vram"),
            "cuda_device": current_profile.get("cuda_device"),
            "fast": current_profile.get("fast"),
            "arguments": _redact_args(comfy_launch_profile_extra_args(current_profile)),
        }
        _write_json(report_dir / "environment.json", environment)

        if pause_managed_comfy is not None:
            pause_token = pause_managed_comfy()
            paused = True
            _log(log, "[이미지 진단] 실행 중이던 관리 Comfy 안전 정지 완료")

        probe_node = _install_probe_node(comfy_root)
        _log(log, "[이미지 진단] 임시 수치 계측 노드 설치 완료")

        baseline_process = ComfyProcess(
            comfy_root=comfy_root,
            python=python,
            cancel_event=cancel_event,
            log=comfy_log,
            extra_args=comfy_launch_profile_extra_args(current_profile),
            verify_manager=False,
            enable_manager=False,
        )
        try:
            if progress:
                progress({"event": "image_diagnostic_prepare", "current": 0, "total": 1})
            baseline_process.start(timeout=900)
            validation = _convert_workflow_once(
                base_url=baseline_process.base_url,
                workflow_path=workflow_path,
                binding=binding,
                cancel_event=cancel_event,
            )
            workflow_summary.update(_workflow_inventory(validation))
        finally:
            baseline_process.stop()
            _copy_process_log(baseline_process, report_dir, "workflow_conversion")

        planned: list[tuple[str, dict[str, Any], int, bool]] = [
            ("current", current_profile, BASELINE_RUNS, False),
            ("highvram_dynamic_on", _profile_with(current_profile, vram_mode="highvram", dynamic_off=False, cpu_vae=False), COMPARISON_RUNS, False),
            ("highvram_dynamic_off", _profile_with(current_profile, vram_mode="highvram", dynamic_off=True, cpu_vae=False), COMPARISON_RUNS, False),
            ("novram_dynamic_on", _profile_with(current_profile, vram_mode="novram", dynamic_off=False, cpu_vae=False), COMPARISON_RUNS, False),
            ("novram_dynamic_off", _profile_with(current_profile, vram_mode="novram", dynamic_off=True, cpu_vae=False), COMPARISON_RUNS, False),
        ]
        seen: set[tuple[tuple[str, ...], bool]] = set()
        for name, profile, limit, unpatched in planned:
            signature = (comfy_launch_profile_extra_args(profile), unpatched)
            if signature in seen:
                continue
            seen.add(signature)
            profiles_result.append(
                _run_profile(
                    comfy_root=comfy_root,
                    python=python,
                    validation=validation,
                    report_dir=report_dir,
                    profile_name=name,
                    profile=profile,
                    run_limit=limit,
                    cancel_event=cancel_event,
                    log=log,
                    comfy_log=comfy_log,
                    progress=progress,
                    unpatched=unpatched,
                )
            )

        if any(profile.get("first_abnormal_run") for profile in profiles_result):
            adaptive = [
                ("highvram_cpu_vae", _profile_with(current_profile, vram_mode="highvram", dynamic_off=True, cpu_vae=True), COMPARISON_RUNS, False),
                ("lowvram_dynamic_off", _profile_with(current_profile, vram_mode="lowvram", dynamic_off=True, cpu_vae=False), COMPARISON_RUNS, False),
                ("core_unpatched", current_profile, COMPARISON_RUNS, True),
            ]
            for name, profile, limit, unpatched in adaptive:
                signature = (comfy_launch_profile_extra_args(profile), unpatched)
                if signature in seen:
                    continue
                seen.add(signature)
                profiles_result.append(
                    _run_profile(
                        comfy_root=comfy_root,
                        python=python,
                        validation=validation,
                        report_dir=report_dir,
                        profile_name=name,
                        profile=profile,
                        run_limit=limit,
                        cancel_event=cancel_event,
                        log=log,
                        comfy_log=comfy_log,
                        progress=progress,
                        unpatched=unpatched,
                    )
                )

        for profile in profiles_result:
            if profile.get("error"):
                errors.append(f"{profile.get('name')}: {profile.get('error')}")
        conclusions = _conclusions(profiles_result)
        result: dict[str, Any] = {
            "operation": "image_diagnostic",
            "diagnostic_id": diagnostic_id,
            "completed_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "duration_seconds": round(time.monotonic() - started, 3),
            "workflow": workflow_summary,
            "environment": environment,
            "profiles": profiles_result,
            "conclusions": conclusions,
            "errors": errors,
            "incomplete": bool(errors),
            "archive_id": diagnostic_id,
            "archive_name": archive_path.name,
        }
        probe_removed = _remove_probe_node(probe_node, comfy_root)
        probe_node = None
        if not probe_removed:
            errors.append("임시 계측 custom node를 완전히 정리하지 못했습니다.")
            result["errors"] = errors
            result["incomplete"] = True
        if paused and resume_managed_comfy is not None:
            resume_managed_comfy(pause_token)
            paused = False
            _log(log, "[이미지 진단] 관리 Comfy 원래 실행 상태 복구 완료")
        _write_json(report_dir / "runs.json", {"profiles": profiles_result})
        _atomic_write(report_dir / "report.md", (_render_report(result) + "\n").encode("utf-8"))
        _make_archive(report_dir, archive_path)
        public = _public_result(result, archive_path)
        _remove_expanded_report(report_dir, root)
        return public
    except ComfyE2ECancelled:
        raise
    except Exception as exc:
        print(f"[COMFY_INSTALL][IMAGE_DIAGNOSTIC] 진단 실행 실패: {exc}")
        traceback.print_exc()
        errors.append(f"{type(exc).__name__}: {exc}")
        probe_removed = _remove_probe_node(probe_node, comfy_root)
        probe_node = None
        if not probe_removed:
            errors.append("임시 계측 custom node를 완전히 정리하지 못했습니다.")
        if paused and resume_managed_comfy is not None:
            try:
                resume_managed_comfy(pause_token)
                paused = False
                _log(log, "[이미지 진단] 관리 Comfy 원래 실행 상태 복구 완료")
            except Exception as resume_exc:
                print(f"[COMFY_INSTALL][IMAGE_DIAGNOSTIC] 관리 Comfy 복구 실패: {resume_exc}")
                traceback.print_exc()
                errors.append(f"관리 Comfy 복구 실패: {type(resume_exc).__name__}: {resume_exc}")
        failure_result = {
            "operation": "image_diagnostic",
            "diagnostic_id": diagnostic_id,
            "completed_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "duration_seconds": round(time.monotonic() - started, 3),
            "workflow": workflow_summary,
            "environment": environment,
            "profiles": profiles_result,
            "conclusions": ["진단을 끝까지 수행하지 못했습니다. 아래 오류와 원본 로그를 확인해야 합니다."],
            "errors": errors,
            "archive_id": diagnostic_id,
            "archive_name": archive_path.name,
            "incomplete": True,
        }
        try:
            _write_json(report_dir / "runs.json", {"profiles": profiles_result, "errors": errors})
            _atomic_write(report_dir / "report.md", (_render_report(failure_result) + "\n").encode("utf-8"))
            _make_archive(report_dir, archive_path)
            public = _public_result(failure_result, archive_path)
            _remove_expanded_report(report_dir, root)
            return public
        except Exception as archive_exc:
            print(f"[COMFY_INSTALL][IMAGE_DIAGNOSTIC] 실패 보고서 ZIP 생성도 실패: {archive_exc}")
            traceback.print_exc()
            raise ImageDiagnosticError(
                f"진단 실패({exc}); 실패 보고서 ZIP 생성 실패({archive_exc})"
            ) from archive_exc
    finally:
        _remove_probe_node(probe_node, comfy_root)
        if paused and resume_managed_comfy is not None:
            try:
                resume_managed_comfy(pause_token)
                _log(log, "[이미지 진단] 관리 Comfy 원래 실행 상태 복구 완료")
            except Exception as exc:
                print(f"[COMFY_INSTALL][IMAGE_DIAGNOSTIC] 관리 Comfy 복구 실패: {exc}")
                traceback.print_exc()
                _log(log, f"[이미지 진단] 관리 Comfy 복구 실패: {exc}", "error")
