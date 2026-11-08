"""Keep llama.cpp resident until ComfyUI next loads an image model on the GPU."""

from functools import wraps
import logging

import comfy.model_management as model_management


def install_vram_handoff(service):
    original = model_management.load_models_gpu
    if getattr(original, "_soya_llamacpp_handoff", False):
        return

    def release():
        if service.is_owned:
            result = service.request_release(source="soya_image_model_load", wait_for_coalesced=True)
            if not result.terminal or not result.success:
                raise RuntimeError(
                    "llama.cpp could not release GPU memory before image model loading: "
                    f"{result.status.value}. Wait for active analysis to finish and retry."
                )
            logging.info("[SOYA Gemma] Image model handoff: %s", result.status.value)

    @wraps(original)
    def load_models_gpu(models, *args, **kwargs):
        models = list(models)
        if any(model.load_device.type != "cpu" for model in models):
            release()
        return original(models, *args, **kwargs)

    load_models_gpu._soya_llamacpp_handoff = True
    model_management.load_models_gpu = load_models_gpu

    # HIGH_VRAM checkpoint loaders allocate weights before load_models_gpu.
    # Release before their initial device selection, including text encoders.
    def before_device_selection(function):
        @wraps(function)
        def select_device(*args, **kwargs):
            if model_management.get_torch_device().type != "cpu":
                release()
            return function(*args, **kwargs)
        return select_device

    for name in (
        "unet_inital_load_device", "unet_offload_device",
        "text_encoder_initial_device", "vae_offload_device",
    ):
        setattr(model_management, name, before_device_selection(getattr(model_management, name)))
