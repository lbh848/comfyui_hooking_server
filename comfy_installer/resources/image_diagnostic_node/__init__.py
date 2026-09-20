from __future__ import annotations

import hashlib
import json
import threading
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image


_RUNS: dict[str, dict] = {}
_LOCK = threading.RLock()


def _tensor_items(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _tensor_items(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _tensor_items(child)


def _tensor_stats(value) -> dict:
    tensors = list(_tensor_items(value))
    if not tensors:
        return {"tensor_count": 0, "nan": 0, "inf": 0, "finite": True}
    count = nan = inf = 0
    total_sum = total_sq = 0.0
    finite_count = 0
    minimum = None
    maximum = None
    signature = hashlib.sha256()
    shapes = []
    dtypes = []
    preview = []
    for tensor in tensors:
        detached = tensor.detach()
        shapes.append(list(detached.shape))
        dtypes.append(str(detached.dtype))
        count += int(detached.numel())
        if detached.numel() == 0:
            continue
        numeric = detached.float()
        finite_mask = torch.isfinite(numeric)
        tensor_nan = int(torch.isnan(numeric).sum().item())
        tensor_inf = int(torch.isinf(numeric).sum().item())
        nan += tensor_nan
        inf += tensor_inf
        finite_values = numeric[finite_mask]
        if finite_values.numel():
            finite_count += int(finite_values.numel())
            finite_double = finite_values.double()
            total_sum += float(finite_double.sum().item())
            total_sq += float((finite_double * finite_double).sum().item())
            local_min = float(finite_values.min().item())
            local_max = float(finite_values.max().item())
            minimum = local_min if minimum is None else min(minimum, local_min)
            maximum = local_max if maximum is None else max(maximum, local_max)
        flattened = torch.nan_to_num(
            numeric.reshape(-1), nan=0.0, posinf=3.4e38, neginf=-3.4e38
        )
        stride = max(int(flattened.numel() // 256), 1)
        sample = flattened[::stride][:256].cpu().numpy().astype(np.float32, copy=False)
        signature.update(sample.tobytes())
        if len(preview) < 32:
            preview.extend(float(value) for value in sample[: 32 - len(preview)])
    mean = total_sum / finite_count if finite_count else None
    variance = max(total_sq / finite_count - mean * mean, 0.0) if finite_count else None
    return {
        "tensor_count": len(tensors),
        "count": count,
        "finite_count": finite_count,
        "nan": nan,
        "inf": inf,
        "finite": nan == 0 and inf == 0,
        "min": minimum,
        "max": maximum,
        "mean": mean,
        "std": variance ** 0.5 if variance is not None else None,
        "sample_sha256": signature.hexdigest(),
        "sample_preview": [round(value, 8) for value in preview],
        "shapes": shapes[:8],
        "dtypes": dtypes[:8],
    }


def _tensor_stats_deferred(value) -> dict:
    """모델 forward 중 CPU 동기화를 피하고 작은 GPU scalar만 보관한다."""
    entries = []
    for tensor in _tensor_items(value):
        detached = tensor.detach()
        count = int(detached.numel())
        if not count:
            entries.append({"count": 0, "shape": list(detached.shape), "dtype": str(tensor.dtype)})
            continue
        flat = detached.reshape(-1)
        stride = max(int(flat.numel() // 256), 1)
        sample = torch.nan_to_num(
            flat[::stride][:256].float(),
            nan=0.0,
            posinf=3.4e38,
            neginf=-3.4e38,
        ).clone()
        floating = torch.is_floating_point(detached) or torch.is_complex(detached)
        zero = torch.zeros((), dtype=torch.int64, device=detached.device)
        entries.append(
            {
                "count": count,
                "shape": list(detached.shape),
                "dtype": str(tensor.dtype),
                "nan_value": torch.isnan(detached).sum() if floating else zero,
                "inf_value": torch.isinf(detached).sum() if floating else zero,
                "sample_value": sample,
            }
        )
    return {"__deferred_tensor_stats__": entries}


def _materialize_deferred_stats(value: dict) -> dict:
    entries = value.get("__deferred_tensor_stats__", [])
    count = nan = inf = finite_count = 0
    samples = []
    signature = hashlib.sha256()
    shapes = []
    dtypes = []
    for entry in entries:
        count += int(entry.get("count", 0))
        shapes.append(entry.get("shape", []))
        dtypes.append(entry.get("dtype"))
        if not entry.get("count"):
            continue
        local_nan = int(entry["nan_value"].item())
        local_inf = int(entry["inf_value"].item())
        nan += local_nan
        inf += local_inf
        sample = entry["sample_value"].cpu().numpy().astype(np.float32, copy=False)
        samples.append(sample)
        signature.update(sample.tobytes())
    finite_count = count - nan - inf
    merged = (
        np.concatenate(samples).astype(np.float64)
        if samples
        else np.asarray([], dtype=np.float64)
    )
    return {
        "tensor_count": len(entries),
        "count": count,
        "finite_count": finite_count,
        "nan": nan,
        "inf": inf,
        "finite": nan == 0 and inf == 0,
        "min": float(merged.min()) if merged.size else None,
        "max": float(merged.max()) if merged.size else None,
        "mean": float(merged.mean()) if merged.size else None,
        "std": float(merged.std()) if merged.size else None,
        "summary_from_sample": True,
        "sample_sha256": signature.hexdigest(),
        "sample_preview": [round(float(value), 8) for value in merged[:32]],
        "shapes": shapes[:8],
        "dtypes": dtypes[:8],
    }


def _materialize(value):
    if isinstance(value, dict):
        if "__deferred_tensor_stats__" in value:
            return _materialize_deferred_stats(value)
        return {key: _materialize(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_materialize(child) for child in value]
    if isinstance(value, tuple):
        return [_materialize(child) for child in value]
    return value


def _run_record(run_id: str) -> dict:
    with _LOCK:
        return _RUNS.setdefault(
            run_id,
            {"model_calls": [], "deep_transitions": [], "errors": []},
        )


def _record_error(run_id: str, stage: str, exc: BaseException) -> None:
    message = f"{type(exc).__name__}: {exc}"
    print(
        "[LB_IMAGE_DIAGNOSTIC] 계측 실패: "
        f"run_id={run_id}, stage={stage}, error={message}"
    )
    traceback.print_exc()
    with _LOCK:
        _run_record(run_id)["errors"].append({"stage": stage, "error": message})


def _deferred_finite_flags(value):
    return [torch.isfinite(tensor.detach()).all() for tensor in _tensor_items(value)]


def _materialize_finite_flags(flags) -> bool:
    return all(bool(flag.item()) for flag in flags)


def _install_transition_hooks(module, observations: list[dict]):
    handles = []
    for name, child in module.named_modules():
        if not name or any(True for _ in child.children()):
            continue

        def hook(current, args, output, *, module_name=name):
            if len(observations) >= 4096:
                return
            try:
                observations.append(
                    {
                        "module": module_name,
                        "module_type": type(current).__name__,
                        "input_flags": _deferred_finite_flags(args),
                        "output_flags": _deferred_finite_flags(output),
                    }
                )
            except Exception as exc:
                print(
                    "[LB_IMAGE_DIAGNOSTIC] 모듈 hook 계측 실패: "
                    f"module={module_name}, error={type(exc).__name__}: {exc}"
                )
                traceback.print_exc()

        handles.append(child.register_forward_hook(hook))
    return handles


class LBDiagnosticModelProbe:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "run_id": ("STRING", {"default": ""}),
                "trace_call": ("INT", {"default": -1, "min": -1, "max": 10000}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "probe"
    CATEGORY = "diagnostic"

    @classmethod
    def IS_CHANGED(cls, model, run_id, trace_call):
        return run_id

    def probe(self, model, run_id, trace_call):
        try:
            patched = model.clone()
            patched.model_options = {**patched.model_options}
            old_wrapper = patched.model_options.get("model_function_wrapper")
            state = {"call": 0}

            def wrapper(apply_model, args):
                call_index = state["call"]
                state["call"] += 1
                handles = []
                observations: list[dict] = []
                if call_index == int(trace_call):
                    try:
                        root = getattr(getattr(patched, "model", None), "diffusion_model", None)
                        if root is not None:
                            handles = _install_transition_hooks(root, observations)
                        else:
                            print(
                                "[LB_IMAGE_DIAGNOSTIC] 정밀 추적 생략: "
                                f"run_id={run_id}, call={call_index}, diffusion_model 없음"
                            )
                    except Exception as exc:
                        _record_error(run_id, "deep_hook_install", exc)
                try:
                    input_value = args.get("input") if isinstance(args, dict) else args
                    timestep = args.get("timestep") if isinstance(args, dict) else None
                    if old_wrapper is not None:
                        output = old_wrapper(apply_model, args)
                    elif isinstance(args, dict):
                        output = apply_model(args["input"], args["timestep"], **args["c"])
                    else:
                        output = apply_model(*args)
                    record = {
                        "call": call_index,
                        "input": _tensor_stats_deferred(input_value),
                        "timestep": _tensor_stats_deferred(timestep),
                        "output": _tensor_stats_deferred(output),
                    }
                    transitions = []
                    for observation in observations:
                        input_finite = _materialize_finite_flags(observation["input_flags"])
                        output_finite = _materialize_finite_flags(observation["output_flags"])
                        if input_finite and not output_finite:
                            transitions.append(
                                {
                                    "module": observation["module"],
                                    "module_type": observation["module_type"],
                                    "input": {"finite": True},
                                    "output": {"finite": False},
                                }
                            )
                            if len(transitions) >= 64:
                                break
                    with _LOCK:
                        current = _run_record(run_id)
                        current["model_calls"].append(record)
                        if transitions:
                            current["deep_transitions"].extend(transitions)
                    return output
                except Exception as exc:
                    _record_error(run_id, f"model_call_{call_index}", exc)
                    raise
                finally:
                    for handle in handles:
                        try:
                            handle.remove()
                        except Exception as exc:
                            _record_error(run_id, "deep_hook_remove", exc)

            patched.set_model_unet_function_wrapper(wrapper)
            return (patched,)
        except Exception as exc:
            _record_error(run_id, "model_probe", exc)
            raise


class LBDiagnosticLatentTrigger:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "nonce": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "trigger"
    CATEGORY = "diagnostic"

    @classmethod
    def IS_CHANGED(cls, latent, nonce):
        return nonce

    def trigger(self, latent, nonce):
        return (latent,)


class LBDiagnosticLatentProbe:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "run_id": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "probe"
    CATEGORY = "diagnostic"

    @classmethod
    def IS_CHANGED(cls, latent, run_id):
        return run_id

    def probe(self, latent, run_id):
        try:
            stats = _tensor_stats(latent.get("samples") if isinstance(latent, dict) else latent)
            with _LOCK:
                _run_record(run_id)["latent"] = stats
            return (latent,)
        except Exception as exc:
            _record_error(run_id, "latent_probe", exc)
            raise


def _image_stats(images: torch.Tensor) -> tuple[dict, np.ndarray]:
    numeric = images.detach().float()
    base = _tensor_stats(numeric)
    clean = torch.nan_to_num(numeric, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    rgb = clean[..., :3]
    luminance = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    horizontal = (
        float(torch.mean(torch.abs(rgb[:, :, 1:, :] - rgb[:, :, :-1, :])).item())
        if rgb.shape[2] > 1 else 0.0
    )
    vertical = (
        float(torch.mean(torch.abs(rgb[:, 1:, :, :] - rgb[:, :-1, :, :])).item())
        if rgb.shape[1] > 1 else 0.0
    )
    channel_mean = rgb.mean(dim=(0, 1, 2)).cpu().tolist()
    channel_std = rgb.std(dim=(0, 1, 2), unbiased=False).cpu().tolist()
    thumb = torch.nn.functional.interpolate(
        rgb.permute(0, 3, 1, 2), size=(16, 16), mode="area"
    )[0].permute(1, 2, 0).reshape(-1).cpu().numpy()
    stats = {
        **base,
        "luminance_mean": float(luminance.mean().item()),
        "luminance_std": float(luminance.std(unbiased=False).item()),
        "black_fraction": float((luminance <= 1.0 / 255.0).float().mean().item()),
        "white_fraction": float((luminance >= 254.0 / 255.0).float().mean().item()),
        "clipped_fraction": float(((rgb <= 0.0) | (rgb >= 1.0)).float().mean().item()),
        "edge_mean": (horizontal + vertical) / 2.0,
        "channel_mean": [round(float(v), 8) for v in channel_mean],
        "channel_std": [round(float(v), 8) for v in channel_std],
        "thumbnail_16x16_rgb": [round(float(v), 6) for v in thumb.tolist()],
    }
    image = (rgb[0].cpu().numpy() * 255.0).round().astype(np.uint8)
    return stats, image


class LBDiagnosticImageProbe:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "run_id": ("STRING", {"default": ""}),
                "profile": ("STRING", {"default": "baseline"}),
                "output_dir": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "measure"
    OUTPUT_NODE = True
    CATEGORY = "diagnostic"

    @classmethod
    def IS_CHANGED(cls, images, run_id, profile, output_dir):
        return run_id

    def measure(self, images, run_id, profile, output_dir):
        try:
            stats, image = _image_stats(images)
            target_dir = Path(output_dir).resolve()
            target_dir.mkdir(parents=True, exist_ok=True)
            safe_profile = "".join(c if c.isalnum() or c in "-_" else "_" for c in profile)
            safe_run = "".join(c if c.isalnum() or c in "-_" else "_" for c in run_id)
            filename = f"{safe_profile}-{safe_run}.png"
            Image.fromarray(image, mode="RGB").save(target_dir / filename)
            with _LOCK:
                record = _materialize(dict(_run_record(run_id)))
                record["image"] = stats
                record["artifact"] = filename
                if torch.cuda.is_available():
                    record["cuda_memory"] = {
                        "allocated": int(torch.cuda.memory_allocated()),
                        "reserved": int(torch.cuda.memory_reserved()),
                        "max_allocated": int(torch.cuda.max_memory_allocated()),
                        "max_reserved": int(torch.cuda.max_memory_reserved()),
                    }
                _RUNS.pop(run_id, None)
            payload = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
            return {"ui": {"diagnostic": [payload], "artifact": [filename]}}
        except Exception as exc:
            _record_error(run_id, "image_probe", exc)
            raise


NODE_CLASS_MAPPINGS = {
    "LBDiagnosticModelProbe": LBDiagnosticModelProbe,
    "LBDiagnosticLatentTrigger": LBDiagnosticLatentTrigger,
    "LBDiagnosticLatentProbe": LBDiagnosticLatentProbe,
    "LBDiagnosticImageProbe": LBDiagnosticImageProbe,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LBDiagnosticModelProbe": "LB Diagnostic Model Probe",
    "LBDiagnosticLatentTrigger": "LB Diagnostic Latent Trigger",
    "LBDiagnosticLatentProbe": "LB Diagnostic Latent Probe",
    "LBDiagnosticImageProbe": "LB Diagnostic Image Probe",
}
