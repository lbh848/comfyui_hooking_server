"""ANIMA reference and regional LoRA preparation for illustration HF/FD/ED."""
import math

import torch
import torch.nn.functional as F
import comfy.utils

from .soya_first_sampler_v2_core import _parse_multi_char, _parse_lora_entries, _resolve_lora_path
from .soya_snapref import SnapRef
from .soya_spatial_lora import apply_spatial_loras


def _snapref(model, clip, vae, images, settings):
    if not images:
        return None
    strength = float(settings.get("strength", 1.0 if settings.get("strength_scale") == "linear" else 0.0))
    if settings.get("strength_scale") != "linear":
        strength = math.exp(strength)
    if not settings.get("enabled") or strength <= 0:
        return None
    return SnapRef(model, clip, vae, images,
                   _resolve_lora_path(settings.get("model", "snapref_v2a_pre2.safetensors")),
                   settings.get("mode", "char"), float(settings.get("megapixels", 1.0)),
                   strength, float(settings.get("fidelity", 1.0)))


def prepare_hires_model(model, clip, vae, positive, image, *, multi, masks,
                        lora_activate, lora_data, images, settings):
    multi = _parse_multi_char(multi)
    if not multi["enable"] and not images:
        return model
    if clip is None:
        raise ValueError("ANIMA HF regional LoRA/reference processing requires CLIP")
    if not multi["enable"]:
        # The first sampler already applied the solo character's global LoRA.
        reference = _snapref(model, clip, vae, {1: next(iter(images.values()))}, settings)
        return apply_spatial_loras(model, [], [torch.ones(1, *image.shape[1:3])],
                                   [positive], positive, snapref=reference)

    names = multi["char_name_list"]
    loaded = []
    for entry in _parse_lora_entries(lora_activate, lora_data, names):
        lora = comfy.utils.load_torch_file(_resolve_lora_path(entry["lora_path"]), safe_load=True)
        loaded.append((names.index(entry["character"]), lora, entry["strength"]))
    conditionings = []
    for index in range(multi["char_num"]):
        parts = (multi["char_trigger_list"][index] + multi["shared_before"]
                 + [multi["char_inform"][index], multi["char_composition_list"][index]] + multi["shared_after"])
        conditionings.append(clip.encode_from_tokens_scheduled(clip.tokenize(", ".join(part for part in parts if part))))
    parts = multi["background_trigger_list"] + multi["shared_before"] + multi["shared_after"]
    if multi["char_num"] > 1 and multi["composition_prompt"] not in parts:
        parts.append(multi["composition_prompt"])
    if multi["background_prompt"] and multi["background_prompt"] not in parts:
        parts.append(multi["background_prompt"])
    background = clip.encode_from_tokens_scheduled(clip.tokenize(", ".join(part for part in parts if part)))
    references = {index + 1: images[name] for index, name in enumerate(names) if name in images}
    reference = _snapref(model, clip, vae, references, settings)
    # HF keeps the supplied RGB ownership at its new resolution. It does not
    # rerun first-pass face tracking or carry its mutable execution state.
    return apply_spatial_loras(model, loaded, masks, conditionings, background, snapref=reference)


def prepare_detailer_model(model, clip, vae, positive, mask, name, images, settings):
    reference = _snapref(model, clip, vae, {1: images[name]} if name in images else {}, settings)
    if reference is None:
        return model
    return apply_spatial_loras(model, [], [mask], [positive], positive, snapref=reference)


def prepare_hand_model(model, vae, positive, mask, crop, image_size, *, clip, masks, names, images, settings):
    x1, y1, x2, y2 = crop
    regions, references = [], {}
    for index, name in enumerate(names):
        if masks:
            full = F.interpolate(masks[index].unsqueeze(1), size=image_size, mode="bilinear", align_corners=False)
            cropped = full[:, :, y1:y2, x1:x2]
            region = F.interpolate(cropped, size=mask.shape[-2:], mode="bilinear", align_corners=False)[:, 0] * mask
        else:
            region = mask
        regions.append(region)
        if name in images and torch.any(region > 0):
            references[index + 1] = images[name]
    reference = _snapref(model, clip, vae, references, settings)
    if reference is None:
        return model
    return apply_spatial_loras(model, [], regions, [positive] * len(regions), positive, snapref=reference)
