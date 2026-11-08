"""Stateless Gemma image analysis with a resident, explicitly owned GGUF runtime."""

import importlib
import json
import logging

import nodes

from .soya_vram_handoff import install_vram_handoff
from .soya_illustration_json_v2 import image_tensor
from .soya_gemma4_context import fit_history


def decode_request(prompt_json):
    payload = json.loads(prompt_json)
    if not isinstance(payload, dict) or payload.get("schema") != "soya.gemma4.vision.v1":
        raise ValueError("Expected a soya.gemma4.vision.v1 JSON request.")
    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string.")
    asset = payload.get("image")
    if asset is None:
        return None, prompt, payload.get("history_ranges", [])
    if not isinstance(asset, dict) or not all(key in asset for key in ("encoding", "data", "sha256")):
        raise ValueError("image requires encoding, data, and sha256.")
    return image_tensor({"assets": {"image": asset}}, "image"), prompt, payload.get("history_ranges", [])


class SoyaGemma4Vision_mdsoya:
    CATEGORY = "Soya/Util"
    FUNCTION = "analyze"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_json": ("STRING", {"multiline": True, "default": ""}),
                "context_size": ("INT", {"default": 16384, "min": 1024, "max": 262144, "step": 256}),
                "max_tokens": ("INT", {"default": 8192, "min": 1, "max": 262144}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647, "control_after_generate": True}),
                "gpu_layers": ("STRING", {"default": "auto"}),
                "gpu_device": ("INT", {"default": 0, "min": 0, "max": 15}),
                "binary_path": ("STRING", {"default": "llama.cpp/b12032-cuda12.4/llama-server.exe"}),
                "port": ("INT", {"default": 8094, "min": 1024, "max": 65535}),
            },
            "optional": {
                "thinking_mode": (["on", "off"], {"default": "on"}),
                "reasoning_budget": (["1024", "2048", "4096"], {"default": "2048"}),
                "preload_only": ("BOOLEAN", {"default": False}),
            },
            "hidden": {"unique_id": "UNIQUE_ID", "dynprompt": "DYNPROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def analyze(self, prompt_json, context_size, max_tokens, temperature, seed,
                gpu_layers, gpu_device, binary_path, port,
                unique_id=None, dynprompt=None, extra_pnginfo=None, thinking_mode="on", reasoning_budget="2048", preload_only=False):
        if str(reasoning_budget) not in ("1024", "2048", "4096"):
            raise ValueError("Gemma reasoning budget must be 1024, 2048, or 4096 tokens")
        image, prompt, history_ranges = (None, "", []) if preload_only else decode_request(prompt_json)
        del prompt_json
        try:
            start_class = nodes.NODE_CLASS_MAPPINGS["StartLlamaCppServer"]
            generate_class = nodes.NODE_CLASS_MAPPINGS["LlamaCppGenerate"]
            budget_class = nodes.NODE_CLASS_MAPPINGS["LlamaCppRequestBudget"]
        except KeyError as exc:
            raise RuntimeError("Install the pinned comfyui-llamacpp dependency and restart ComfyUI.") from exc
        package = start_class.__module__.rsplit(".nodes.", 1)[0]
        service = importlib.import_module(f"{package}.runtime.service").get_runtime_service()
        install_vram_handoff(service)
        starter = importlib.import_module(start_class.__module__)
        catalog = importlib.import_module(f"{package}.models.catalog").ModelCatalog()
        resolution = starter.resolve_direct_projector(
            "gemma4-12b-uncensored/gemma-4-12b-it-uncensored-Q4_K_M.gguf",
            "gemma4-12b-uncensored/mmproj-gemma-4-12B-it-bf16.gguf",
        )
        draft_path = catalog.resolve("gemma4-12b-uncensored/mtp-gemma-4-12b-it-uncensored.gguf")
        # Typed argv preserves model paths containing spaces on Windows.
        config = starter.ServerConfig(
            model_path=str(resolution.model_path), mmproj_path=str(resolution.projector_path),
            context_size=context_size, n_gpu_layers=starter.parse_gpu_layers(gpu_layers), main_gpu=0,
            port=port, batch_size=256, flash_attention_mode="on", fit=True,
            extra_args=("--device", f"CUDA{gpu_device}", "--split-mode", "none", "--fit-target", "1536",
                        "--parallel", "1", "--jinja", "--reasoning-budget", str(reasoning_budget),
                        "--model-draft", str(draft_path), "--spec-type", "draft-mtp",
                        "--spec-draft-n-max", "2", "--spec-draft-ngl", "auto"),
        )
        manager = starter.get_server_manager()
        success, error = manager.start(
            config, timeout=180, binary_path=starter.optional_path(binary_path),
            unload_comfy_models_before_start=True, projector_status=resolution.status_dict(),
        )
        if not success:
            raise RuntimeError(f"Gemma runtime could not start: {error}")
        if preload_only:
            return {"ui": {"text": ["공방-로컬 LLM 로드 완료"]}, "result": ("공방-로컬 LLM 로드 완료",)}
        server_url = manager.server_url
        options = dict(
            server_url=server_url, image_1=image, image_amount=0 if image is None else 1,
            include_image_batch=True, thinking_mode=thinking_mode, max_tokens=max_tokens,
            sampling_mode="custom", temperature=temperature, top_p=0.95, top_k=64,
            min_p=0.0, repeat_penalty=1.0, seed=seed,
            cache_prompt=False, release_after_generation=False, request_timeout=600,
        )
        def count_budget(text):
            return budget_class().generate(prompt=text, budget_policy="report", **options)["result"][2].as_dict()

        prompt, context_budget = fit_history(prompt, history_ranges, count_budget)
        if context_budget["history_turns_removed"]:
            logging.info("[SOYA Gemma] History fitted to context: %s", context_budget)
        output = generate_class().generate(
            prompt=prompt, **options, budget_policy="enforce",
            unique_id=unique_id, dynprompt=dynprompt, extra_pnginfo=extra_pnginfo,
        )
        response, thinking, result = output["result"]
        return {
            "ui": {**output["ui"], "soya_gemma4": [{
                "reasoning_text": thinking, "usage": result.usage.as_dict(),
                "finish_reason": result.finish_reason,
                "context_budget": context_budget,
            }]},
            "result": (response,),
        }


def register_gemma4_vision(class_mappings, display_name_mappings):
    class_mappings["SoyaGemma4Vision_mdsoya"] = SoyaGemma4Vision_mdsoya
    display_name_mappings["SoyaGemma4Vision_mdsoya"] = "Gemma 4 Image Analyze (Soya)"
