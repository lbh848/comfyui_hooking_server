"""JSON-consuming V2 nodes. Each node reads its own keys from the same text.

Existing node classes and IDs retain their behavior. Models, samplers and
detailers are reused; only the V2 input contract and in-memory asset I/O differ.
"""
import base64
import copy
import hashlib
import importlib
import io
import json
import logging
import math
from pathlib import Path

from PIL import Image, ImageOps


# These nodes prepare reusable models or conditioning; none renders an image.
# Keep reuse inside the V2 wrappers so the standard Comfy cache stays untouched.
REUSABLE_BASES = frozenset({
    "CheckpointLoaderKJ", "UNETLoader", "CLIPLoader", "VAELoader",
    "UltralyticsDetectorProvider", "SoyaBatchLoraLoader_mdsoya",
    "PathchSageAttentionKJ", "ModelPatchTorchSettings",
    "SkimmedCFG_LinInterp_CFG_PreCFG", "DCWModelPatch",
    "CLIPSetLastLayer", "CLIPTextEncode",
})


def resolved_input_signature(value):
    """Compare extracted values, retaining no reference to the JSON document."""
    import torch
    if value is None or isinstance(value, (str, bool, int, float)):
        return (type(value), value)
    if isinstance(value, dict):
        return (dict, tuple((key, resolved_input_signature(item)) for key, item in sorted(value.items())))
    if isinstance(value, (tuple, list)):
        return (type(value), tuple(resolved_input_signature(item) for item in value))
    # Comfy objects are passed by reference. Patcher revisions also distinguish
    # in-place patch changes; tensors distinguish in-place writes.
    return (type(value), id(value), getattr(value, "patches_uuid", None),
            value._version if isinstance(value, torch.Tensor) and not value.is_inference() else None)


def read_payload(prompt_json):
    payload = json.loads(prompt_json)
    if not isinstance(payload, dict) or payload.get("schema") != "soya.illustration.v2":
        raise ValueError("Expected a soya.illustration.v2 JSON prompt")
    if not isinstance(payload.get("fields"), dict) or not isinstance(payload.get("assets"), dict):
        raise ValueError("Illustration JSON requires fields and assets objects")
    return payload


def asset_bytes(payload, key):
    asset = payload["assets"][key]
    if asset.get("encoding") != "base64":
        raise ValueError(f"Unsupported asset encoding: {key}")
    data = base64.b64decode(asset["data"], validate=True)
    if hashlib.sha256(data).hexdigest() != asset["sha256"]:
        raise ValueError(f"Illustration asset checksum mismatch: {key}")
    return data


def image_tensor(payload, key):
    import numpy as np
    import torch
    with Image.open(io.BytesIO(asset_bytes(payload, key))) as source:
        rgb = np.array(ImageOps.exif_transpose(source).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(rgb).unsqueeze(0)


def mask_tensor(payload, key):
    import numpy as np
    import torch
    with Image.open(io.BytesIO(asset_bytes(payload, key))) as source:
        gray = np.array(ImageOps.exif_transpose(source).convert("L"), dtype=np.float32) / 255.0
    return torch.from_numpy(gray).unsqueeze(0)


def asset_key(payload, field):
    key = payload["fields"].get(field)
    if not isinstance(key, str) or key not in payload["assets"]:
        raise ValueError(f"Inpainting JSON requires a base64 asset referenced by fields.{field}")
    return key


class SoyaInpaintJSONV2_mdsoya:
    CATEGORY = "Soya/Illustration V2"
    FUNCTION = "resolve"
    RETURN_TYPES = ("STRING", "STRING", "IMAGE", "MASK", "INT", "INT", "FLOAT", "FLOAT", "INT", "FLOAT")
    RETURN_NAMES = ("EDIT_PROMPT", "NEGATIVE_PROMPT", "SOURCE_IMAGE", "MASK", "SEED", "STEPS", "CFG", "DENOISE", "MASK_GROW", "MASK_BLUR")
    DESCRIPTION = "Decodes the source and mask assets from the shared JSON prompt for inpainting."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"prompt_json": ("STRING", {"forceInput": True})}}

    def resolve(self, prompt_json):
        payload = read_payload(prompt_json)
        source = image_tensor(payload, asset_key(payload, "SOURCE_ASSET"))
        mask = mask_tensor(payload, asset_key(payload, "MASK_ASSET"))
        if mask.shape[-2:] != source.shape[1:3]:
            raise ValueError("Inpainting source and mask sizes differ")
        return (
            field_value(payload, "EDIT_PROMPT", "STRING"),
            field_value(payload, "NEGATIVE_PROMPT", "STRING"),
            source,
            mask,
            field_value(payload, "SEED", "INT"),
            field_value(payload, "STEPS", "INT"),
            field_value(payload, "CFG", "FLOAT"),
            field_value(payload, "DENOISE", "FLOAT"),
            field_value(payload, "MASK_GROW", "INT"),
            field_value(payload, "MASK_BLUR", "FLOAT"),
        )


def regional_masks(payload):
    fields = payload["fields"]
    multi = fields.get("MULTI_CHAR", {})
    if not multi.get("enable"):
        return []
    count = int(multi["char_num"])
    if not 1 <= count <= 3:
        raise ValueError("Regional RGB masks support one to three characters")
    rgb = image_tensor(payload, fields["REGION_MASK"])
    masks = [rgb[:, :, :, index].contiguous() for index in range(count)]
    for index, channel in enumerate(masks):
        if float(channel.max()) <= 0:
            raise ValueError(f"Regional mask channel {'RGB'[index]} is empty")
    return masks


def stage_reference_assets(payload, stage):
    """Select reference asset IDs for both execution and flow previews."""
    settings = payload["fields"].get("SNAPREF", {})
    if (not settings.get("enabled")
            or settings.get("strength_scale") == "linear" and float(settings.get("strength", 1.0)) <= 0):
        return {}
    grouped = {}
    for item in settings.get("references", []):
        grouped.setdefault(item["CHAR"], []).append(item)
    images = {}
    for name, refs in grouped.items():
        general, faces = [], []
        for index, item in enumerate(refs):
            # Older saved JSON represents a secondary face as a plain asset.
            is_face = item.get("role") == "face" or (index == 1 and "role" not in item
                       and settings.get("secondary_reference") == "face")
            if item.get("face_asset"):
                faces.append(item["face_asset"])
            (faces if is_face else general).append(item["asset"])
        mode = settings.get("secondary_reference", "none")
        if mode == "face_only":
            keys = faces[:1] if faces else general[:1]
        elif mode == "stage":
            keys = faces[:1] if stage in {"face", "eye"} and faces else general[:1]
        elif mode == "face":
            keys = general[:1] + faces[:1]
        elif mode == "representative":
            keys = general[:2]
        else:
            keys = general[:1]
        if keys:
            images[name] = list(dict.fromkeys(keys))
    return images


def stage_reference_images(payload, stage):
    return {name: [image_tensor(payload, key) for key in keys]
            for name, keys in stage_reference_assets(payload, stage).items()}


def field_value(payload, key, dtype):
    value = payload["fields"][key]
    if dtype == "STRING":
        if isinstance(value, bool):
            return "true" if value else "false"
        if key == "CHAR_LIST":
            return ",".join(value)
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        return str(value)
    if dtype == "INT":
        if isinstance(value, bool) or int(value) != value:
            raise ValueError(f"{key} must be an integer")
        return int(value)
    if dtype == "FLOAT":
        return float(value)
    return value


def sampling_values(payload, sampling_key, sampling, declarations):
    settings = payload["fields"].get(sampling_key)
    if not isinstance(settings, dict):
        raise ValueError(f"JSON prompt requires a settings object at fields.{sampling_key}")
    result = {}
    for name in sampling["inputs"]:
        path = f"fields.{sampling_key}.{name}"
        if name not in settings:
            raise ValueError(f"JSON prompt is missing {path}")
        declaration = declarations[name]
        dtype = declaration[0]
        options = declaration[1] if len(declaration) > 1 else {}
        value = settings[name]
        if isinstance(dtype, list):
            if value not in dtype:
                raise ValueError(f"{path} has an unsupported value: {value!r}")
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{path} must be a number")
            if not math.isfinite(value):
                raise ValueError(f"{path} must be finite")
            if dtype == "INT" and int(value) != value:
                raise ValueError(f"{path} must be an integer")
            value = field_value({"fields": settings}, name, dtype)
            if "min" in options and value < options["min"]:
                raise ValueError(f"{path} must be at least {options['min']}")
            if "max" in options and value > options["max"]:
                raise ValueError(f"{path} must be at most {options['max']}")
        result[name] = value
    return result


class SoyaModelPreload_mdsoya:
    CATEGORY = "Soya/Util"
    FUNCTION = "load"
    RETURN_TYPES = ()
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}, "optional": {"secondary_model": ("MODEL",)}}

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def load(self, model, secondary_model=None):
        import comfy.model_management as model_management
        import comfy.ops

        models = [model] + ([secondary_model] if secondary_model is not None else [])
        model_management.load_models_gpu(models)
        loaded = []
        for patcher in models:
            device = patcher.load_device
            if model_management.is_device_cpu(device):
                raise RuntimeError("그림 모델의 로드 장치가 CPU입니다. GPU 실행 설정을 확인하세요.")
            if patcher.is_dynamic():
                # Dynamic load_models_gpu only stages weights in host memory.
                # Touch each layer through the same cast/cache path used by
                # inference, releasing its pin while retaining the VRAM cache.
                dtype = getattr(patcher.model, "manual_cast_dtype", None) or patcher.model_dtype()
                for module in patcher.model.modules():
                    if not hasattr(module, "_v"):
                        continue
                    model_management.throw_exception_if_processing_interrupted()
                    with comfy.ops.CastBiasWeightContext(module, dtype=dtype, device=device, offloadable=True):
                        pass
                model_management.synchronize()
            resident_bytes = patcher.loaded_size()
            if resident_bytes <= 0:
                raise RuntimeError("그림 모델의 GPU 메모리 로드를 확인하지 못했습니다.")
            loaded.append({"model": type(patcher.model).__name__, "resident_bytes": resident_bytes})
            logging.info("[SOYA Model Preload] %s GPU resident: %.1f MiB", loaded[-1]["model"], resident_bytes / (1024 ** 2))
        return {"ui": {"text": ["그림 모델 GPU 로드 완료"], "soya_model_preload": loaded}, "result": ()}


class SoyaRenderTargetSelect_mdsoya:
    """Switch the decode target without rebuilding the shared diffusion model."""

    CATEGORY = "Soya/Illustration V2"
    FUNCTION = "select"
    RETURN_TYPES = ("MODEL", "VAE", "STRING")
    RETURN_NAMES = ("model", "vae", "sampler_prompt_json")
    DESCRIPTION = ("Reads fields.RENDER_TARGET and returns the character or background "
                   "model, decode VAE and the sampler's JSON prompt.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "prompt_json": ("STRING", {"forceInput": True}),
            "model_character": ("MODEL",),
            "model_background": ("MODEL",),
            "vae_character": ("VAE",),
            "vae_background": ("VAE",),
        }}

    def select(self, prompt_json, model_character, model_background, vae_character, vae_background):
        payload = read_payload(prompt_json)
        target = str(payload["fields"].get("RENDER_TARGET", "character")).strip().lower()
        if target not in {"character", "background"}:
            raise ValueError(f"RENDER_TARGET must be character or background: {target!r}")
        # Both branches applied their own LoRA set before this node.
        payload["fields"]["LORA_ACTIVATE"] = False
        sampler_json = json.dumps(payload, ensure_ascii=False)
        if target == "background":
            return model_background, vae_background, sampler_json
        return model_character, vae_character, sampler_json


def register_illustration_v2(class_mappings, display_mappings):
    class_mappings["SoyaRenderTargetSelect_mdsoya"] = SoyaRenderTargetSelect_mdsoya
    display_mappings["SoyaRenderTargetSelect_mdsoya"] = "Render Target Select (Soya JSON)"
    class_mappings["SoyaInpaintJSONV2_mdsoya"] = SoyaInpaintJSONV2_mdsoya
    display_mappings["SoyaInpaintJSONV2_mdsoya"] = "Inpainting Inputs V2 (Soya JSON)"
    class_mappings["SoyaModelPreload_mdsoya"] = SoyaModelPreload_mdsoya
    display_mappings["SoyaModelPreload_mdsoya"] = "Load Image Model (Soya)"
    specs = json.loads(Path(__file__).with_name("illustration_v2_specs.json").read_text(encoding="utf-8"))
    # Original SOYA classes are available here. Other packages finish loading
    # before INPUT_TYPES or execution resolves their registered classes.
    originals = dict(class_mappings)

    def make_node(spec):
        sampling = spec.get("sampling")
        model = spec.get("model")
        reuse_outputs = spec["base"] in REUSABLE_BASES

        def base_class():
            if spec["base"] in originals:
                base = originals[spec["base"]]
            else:
                import nodes
                base = nodes.NODE_CLASS_MAPPINGS[spec["base"]]
            core = {"SoyaFirstSampler_mdsoya": "soya_first_sampler_v2_core",
                    "SoyaIPAPatchMaker_mdsoya": "soya_face_match_v2_core",
                    "SoyaCharLoraFaceDetailer_mdsoya": "soya_char_lora_face_detailer_v2_core",
                    "SoyaCharLoraEyeDetailer_mdsoya": "soya_char_lora_eye_detailer_v2_core",
                    "SoyaHandDetailerToggle_mdsoya": "soya_hand_detailer_toggle_v2_core"}.get(spec["base"])
            if core:
                module = importlib.import_module(base.__module__.rsplit(".", 1)[0] + "." + core)
                return getattr(module, spec["base"])
            return base

        class JsonNode:
            RETURN_TYPES = tuple(spec["output"])
            RETURN_NAMES = tuple(spec["output_name"])
            OUTPUT_IS_LIST = tuple(spec["output_is_list"])
            INPUT_IS_LIST = spec.get("input_is_list", False)
            OUTPUT_NODE = spec.get("output_node", False)
            FUNCTION = "execute"
            CATEGORY = "Soya/Illustration V2"
            DESCRIPTION = "Reads its own fields from the shared illustration JSON prompt."

            @classmethod
            def INPUT_TYPES(cls):
                result = copy.deepcopy(base_class().INPUT_TYPES())
                for group in ("required", "optional"):
                    for name in list(result.get(group, {})):
                        if (name in spec["json_inputs"] or name in spec.get("fixed_inputs", {})
                                or sampling and name in sampling["inputs"]
                                or model and name == model["input"]):
                            result[group].pop(name)
                result.setdefault("required", {})["prompt_json"] = ("STRING", {"forceInput": True})
                if spec["base"] == "CLIPTextEncode":
                    result.setdefault("optional", {})["json_key"] = ("STRING", {
                        "default": spec["json_inputs"]["text"]["key"],
                        "multiline": False,
                        "tooltip": "JSON field to encode, e.g. ANIMA_CONTENT, SDXL_CONTENT or NEGATIVE_ANIMA. "
                                   "Separate keys with commas to combine them in order, e.g. ANIMA_QUALITY, ANIMA_CONTENT. "
                                   "The fields. prefix is optional for each key.",
                    })
                if sampling:
                    result.setdefault("optional", {})["sampling_key"] = ("STRING", {
                        "default": sampling["key"], "multiline": False,
                        "tooltip": "JSON settings object under fields. Reads " + ", ".join(sampling["inputs"])
                                   + ". Choose the ANIMA or ILXL block for this stage.",
                    })
                if model:
                    result.setdefault("optional", {})["model_key"] = ("STRING", {
                        "default": model["key"], "multiline": False,
                        "tooltip": "JSON field containing the installed model filename, e.g. " + model["key"]
                                   + ". The fields. prefix is optional.",
                    })
                if spec["base"] == "SoyaFirstSampler_mdsoya":
                    # Keep existing saved widget positions, including sampling_key.
                    optional = result["optional"]
                    optional["snapref_fidelity"] = optional.pop("snapref_fidelity")
                if spec["base"] == "SoyaHiresfixToggle_mdsoya" and sampling["key"] == "ANIMA_HIRESFIX":
                    result.setdefault("optional", {})["clip"] = ("CLIP",)
                return result

            def prepare_inputs(self, prompt_json, kwargs):
                # Keep the large parsed payload in this short-lived frame only.
                if self.INPUT_IS_LIST:
                    prompt_json = prompt_json[0]
                json_keys = None
                if spec["base"] == "CLIPTextEncode":
                    json_key = kwargs.pop("json_key", spec["json_inputs"]["text"]["key"])
                    if self.INPUT_IS_LIST:
                        json_key = json_key[0]
                    json_keys = [key.strip().removeprefix("fields.") for key in json_key.split(",")]
                    if not all(json_keys):
                        raise ValueError("json_key must contain non-empty field names, e.g. ANIMA_QUALITY, ANIMA_CONTENT")
                payload = read_payload(prompt_json)
                resolved = dict(spec.get("fixed_inputs", {}))
                if model:
                    model_key = kwargs.pop("model_key", model["key"])
                    if self.INPUT_IS_LIST:
                        model_key = model_key[0]
                    model_key = model_key.strip().removeprefix("fields.")
                    if not model_key:
                        raise ValueError("model_key must name a JSON field containing a model filename")
                    filename = payload["fields"].get(model_key)
                    if not isinstance(filename, str) or not filename.strip():
                        raise ValueError(f"JSON prompt requires a model filename at fields.{model_key}")
                    filename = filename.strip()
                    schema = base_class().INPUT_TYPES()
                    declarations = {**schema.get("required", {}), **schema.get("optional", {})}
                    if filename not in declarations[model["input"]][0]:
                        raise ValueError(f"Model {filename!r} from fields.{model_key} is not available for {spec['base']}")
                    resolved[model["input"]] = filename
                for name, binding in spec["json_inputs"].items():
                    if name == "text" and json_keys is not None:
                        values = []
                        for key in json_keys:
                            if key not in payload["fields"]:
                                raise ValueError(f"JSON prompt is missing fields.{key}; check json_key on this CLIP Text Encode node")
                            values.append(field_value(payload, key, binding["type"]))
                        resolved[name] = ", ".join(value for value in values if value)
                    else:
                        resolved[name] = field_value(payload, binding["key"], binding["type"])
                if sampling:
                    sampling_key = kwargs.pop("sampling_key", sampling["key"])
                    if self.INPUT_IS_LIST:
                        sampling_key = sampling_key[0]
                    sampling_key = sampling_key.strip().removeprefix("fields.")
                    if not sampling_key:
                        raise ValueError("sampling_key must name a JSON settings object")
                    schema = base_class().INPUT_TYPES()
                    declarations = {**schema.get("required", {}), **schema.get("optional", {})}
                    resolved.update(sampling_values(payload, sampling_key, sampling, declarations))
                if self.INPUT_IS_LIST:
                    resolved = {key: [value] for key, value in resolved.items()}
                kwargs.update(resolved)
                fields = payload["fields"]
                if spec["base"] == "SoyaFirstSampler_mdsoya":
                    multi = fields.get("MULTI_CHAR", {})
                    masks = regional_masks(payload)
                    refs = fields.get("SNAPREF", {}).get("references", [])
                    names = multi.get("char_name_list", []) if multi.get("enable") else list(dict.fromkeys(item["CHAR"] for item in refs))
                    if not multi.get("enable") and len(names) > 1:
                        raise ValueError("Multiple SnapRef references require regional RGB masks")
                    by_name = stage_reference_images(payload, "first")
                    images = {index + 1: by_name[name] for index, name in enumerate(names) if name in by_name}
                    kwargs.update(_embedded_region_masks=masks, _embedded_reference_images=images)
                elif spec["base"] in {"SoyaHiresfixToggle_mdsoya", "SoyaHiresfixSpectrumToggle_mdsoya"} and sampling["key"] == "ANIMA_HIRESFIX":
                    if str(kwargs["enable"]).strip().lower() in {"true", "1", "yes", "on"}:
                        kwargs["_hires_regions"] = {
                            "multi": fields.get("MULTI_CHAR", {}), "masks": regional_masks(payload),
                            "lora_activate": fields.get("LORA_ACTIVATE", False),
                            "lora_data": fields.get("LORA_DATA", {"list": []}),
                            "images": stage_reference_images(payload, "hires"),
                            "settings": {key: value for key, value in fields.get("SNAPREF", {}).items() if key != "references"},
                        }
                elif spec["base"] in {"SoyaCharLoraFaceDetailer_mdsoya", "SoyaCharLoraEyeDetailer_mdsoya"}:
                    if kwargs.get("base_model", "").strip() == "anima" and str(kwargs["enable"]).strip().lower() in {"true", "1", "yes"}:
                        stage = "face" if spec["base"] == "SoyaCharLoraFaceDetailer_mdsoya" else "eye"
                        kwargs["_reference_images"] = stage_reference_images(payload, stage)
                        kwargs["_reference_settings"] = {key: value for key, value in fields.get("SNAPREF", {}).items() if key != "references"}
                elif spec["base"] == "SoyaHandDetailerToggle_mdsoya" and sampling["key"] == "ANIMA_HAND_DETAILER":
                    if str(kwargs["enable"]).strip().lower() in {"true", "1", "yes", "on"}:
                        images = stage_reference_images(payload, "hand")
                        if images:
                            multi = fields.get("MULTI_CHAR", {})
                            kwargs["_hand_references"] = {
                                "masks": regional_masks(payload),
                                "names": multi["char_name_list"] if multi.get("enable") else list(images),
                                "images": images,
                                "settings": {key: value for key, value in fields.get("SNAPREF", {}).items() if key != "references"},
                            }
                elif spec["base"] == "SoyaIPAPatchMaker_mdsoya":
                    import torch
                    entries = fields.get("FACE_EMBEDDINGS", [])
                    kwargs["embed_cache_data"] = json.dumps({"list": [
                        {"CHAR": entry["CHAR"], "emb_path": entry["asset"]} for entry in entries]})
                    kwargs["_embedded_caches"] = {
                        entry["CHAR"]: torch.load(io.BytesIO(asset_bytes(payload, entry["asset"])),
                                                  map_location="cpu", weights_only=True)
                        for entry in entries
                    }
                del payload, fields
                return kwargs

            def execute(self, prompt_json, **kwargs):
                kwargs = self.prepare_inputs(prompt_json, kwargs)
                # The graph owns the shared source; this node no longer needs it.
                del prompt_json
                regions = kwargs.pop("_hires_regions", None)
                if regions is not None:
                    from .soya_illustration_stages import prepare_hires_model
                    kwargs["model"] = prepare_hires_model(kwargs["model"], kwargs.get("clip"), kwargs["vae"],
                                                          kwargs["positive"], kwargs["image"], **regions)
                if spec["base"] == "SoyaHiresfixToggle_mdsoya":
                    kwargs.pop("clip", None)
                hand = kwargs.pop("_hand_references", None)
                if hand is not None:
                    from functools import partial
                    from .soya_illustration_stages import prepare_hand_model
                    kwargs["_reference_model"] = partial(prepare_hand_model, clip=kwargs["clip"], **hand)
                if not hasattr(self, "_implementation"):
                    self._implementation = base_class()()
                signature = resolved_input_signature(kwargs) if reuse_outputs else None
                if reuse_outputs:
                    cached = getattr(self, "_resolved_cache", None)
                    if cached is not None and cached[0] == signature:
                        logging.info("[Soya:JSONReuse] %s: reuse", spec["name"])
                        return cached[2]
                    # One entry per node instance. Release the previous inputs
                    # and result before loading a different model/configuration.
                    self._resolved_cache = None
                    del cached
                result = getattr(self._implementation, self._implementation.FUNCTION)(**kwargs)
                if reuse_outputs:
                    # Retain only this entry's resolved arguments/model refs,
                    # never prompt_json or the parsed Base64 asset document.
                    self._resolved_cache = (signature, kwargs, result)
                    logging.info("[Soya:JSONReuse] %s: updated", spec["name"])
                return result

        if spec["base"] == "SoyaFirstSampler_mdsoya":
            # Queueing the same fixed seed explicitly requests another sample.
            # Only this node and its descendants are invalidated, not loaders.
            JsonNode.IS_CHANGED = classmethod(lambda cls, **kwargs: float("nan"))
        JsonNode.__name__ = spec["name"]
        return JsonNode

    for name, spec in specs.items():
        class_mappings[name] = make_node(spec)
        display_mappings[name] = spec["display_name"]
    from .soya_spatial_face_context import NODE_NAME, SoyaSpatialFaceContextV2
    class_mappings[NODE_NAME] = SoyaSpatialFaceContextV2
    display_mappings[NODE_NAME] = "Spatial Face Context V2 (no identity cache)"
