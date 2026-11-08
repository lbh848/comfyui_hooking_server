"""Self-contained JSON transport for the illustration V2 workflows.

The editor's tagged draft format is accepted at the server boundary only.
Comfy nodes receive one JSON document, including every required input asset.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
import math
import mimetypes
import re
from pathlib import Path

from modes.prompt_path_normalizer import _CONTROL_BLOCK_HEADER_RE, _normalize_payload_paths

SCHEMA = "soya.illustration.v2"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
JSON_FIELDS = {"CACHE_PATH", "FACE_ID_DIR", "LORA_DATA", "FACE_LORA_DATA",
               "STYLE_LORA_DATA", "CHAR_FACE_TAG_INFORM", "MULTI_CHAR", "SNAPREF"}
JSON_FIELDS.update({"ANIMA_FIRST_SAMPLER", "ANIMA_HIRESFIX", "ILXL_HIRESFIX"} | {
    f"{model}_{stage}_DETAILER" for model in ("ANIMA", "ILXL") for stage in ("FACE", "HAND", "EYE")
})
FLOAT_FIELDS = {"FACE_CROP_TOP", "FACE_CROP_BOTTOM", "HRF_SIZE", "FACE_ID_STR"}
INT_FIELDS = {"IMG_W", "IMG_H", "SEED"}
FIRST_SAMPLER_MODES = ("KSampler", "FAST", "SpectrumKSamplerModGuidance")
FIRST_SAMPLER_NUMERIC_CONTROLS = {"spd_scale": (0.25, 1.0), "spd_sigma": (0.0, 1.0)}
FIRST_SAMPLER_CONTROLS = ("sampler_mode", "sage_attention", *FIRST_SAMPLER_NUMERIC_CONTROLS)
REFERENCE_PREPROCESS_LABELS = {
    "none": "전처리 없음", "lineart": "선만 따기(흑백)",
    "lineart_color": "선만 따기(원본 색상)", "grayscale": "흑백화",
}


def preprocess_reference_images(payload: dict) -> None:
    """Transform embedded references, retaining originals for later edits/replay."""
    settings = payload["fields"].get("SNAPREF", {})
    if not settings.get("enabled") or (settings.get("strength_scale") == "linear"
                                       and float(settings.get("strength", 1.0)) <= 0):
        return
    mode = settings.get("preprocess", "none")
    if mode not in REFERENCE_PREPROCESS_LABELS:
        raise ValueError(f"지원하지 않는 레퍼런스 전처리입니다: {mode}")
    assets = payload["assets"]
    processed = {}
    for reference in settings.get("references", []):
        for field in ("asset", "face_asset"):
            if not reference.get(field):
                continue
            source_field = "source_asset" if field == "asset" else "face_source_asset"
            source_key = reference.get(source_field, reference[field])
            if mode == "none":
                reference[field] = source_key
                continue
            reference[source_field] = source_key
            if source_key not in processed:
                import numpy as np
                from PIL import Image, ImageFilter, ImageOps

                original = assets[source_key]
                data = base64.b64decode(original["data"], validate=True)
                if hashlib.sha256(data).hexdigest() != original["sha256"]:
                    raise ValueError(f"레퍼런스 이미지 체크섬이 일치하지 않습니다: {source_key}")
                with Image.open(io.BytesIO(data)) as source:
                    rgba = ImageOps.exif_transpose(source).convert("RGBA")
                    image = Image.new("RGBA", rgba.size, "white")
                    image.alpha_composite(rgba)
                    image = image.convert("RGB")
                if mode == "grayscale":
                    image = ImageOps.grayscale(image).convert("RGB")
                else:
                    # Local dark strokes survive; uniform fills become white.
                    # Per-channel contrast also retains isoluminant colored ink.
                    radius = max(1, min(4, round(min(image.size) / 512)))
                    pixels = np.asarray(image, dtype=np.float32)
                    background = np.asarray(image.filter(ImageFilter.MaxFilter(radius * 2 + 1)), dtype=np.float32)
                    contrast = np.max(background - pixels, axis=2)
                    ink = np.clip((contrast - 8.0) / 48.0, 0, 1)[..., None]
                    color = pixels if mode == "lineart_color" else np.zeros_like(pixels)
                    image = Image.fromarray(np.rint(255 * (1 - ink) + color * ink).astype(np.uint8))
                output = io.BytesIO()
                image.save(output, format="PNG")
                data = output.getvalue()
                digest = hashlib.sha256(data).hexdigest()
                assets.setdefault(digest, {"encoding": "base64", "name": f"{Path(original['name']).stem}_{mode}.png",
                                          "media_type": "image/png", "sha256": digest,
                                          "data": base64.b64encode(data).decode("ascii")})
                processed[source_key] = digest
            reference[field] = processed[source_key]


def workflow_source_resolver(workflow: dict):
    """Resolve the existing UI Set/Get routes without converting or running nodes."""
    nodes = {node["id"]: node for node in workflow["nodes"]}
    links = {link[0]: link for link in workflow["links"]}
    setters = {node["widgets_values"][0]: node for node in nodes.values() if node["type"] == "SetNode"}

    def resolve(identity, slot, seen=None):
        seen = set() if seen is None else seen
        if identity in seen:
            raise ValueError("Cyclic Set/Get route")
        seen.add(identity)
        node = nodes[identity]
        if node["type"] == "GetNode":
            node = setters[node["widgets_values"][0]]
        if node["type"] == "SetNode":
            link = links[node["inputs"][0]["link"]]
            return resolve(link[1], link[2], seen)
        return identity, slot

    return resolve


def workflow_json_settings_nodes(workflow: dict) -> dict:
    """Project UI widgets onto the same input contract used by API workflows."""
    if not isinstance(workflow.get("nodes"), list):
        return workflow
    links = {link[0]: link for link in workflow.get("links", [])}
    resolve = workflow_source_resolver(workflow)
    result = {}
    for node in workflow["nodes"]:
        inputs = {}
        widgets = [item for item in node.get("inputs", []) if item.get("widget")]
        values = node.get("widgets_values", [])
        for selector in ("sampling_key", "model_key", *FIRST_SAMPLER_CONTROLS):
            position = next((i for i, item in enumerate(widgets) if item["name"] == selector), None)
            if position is not None:
                if len(widgets) != len(values):
                    raise ValueError(f"Cannot map workflow widgets: node {node['id']}")
                inputs[selector] = values[position]
        for item in node.get("inputs", []):
            if (item["name"] == "prompt_json" or item.get("type") == "MODEL") and item.get("link") in links:
                link = links[item["link"]]
                identity, slot = resolve(link[1], link[2])
                inputs[item["name"]] = [str(identity), slot]
        if node.get("title") == "긍정프롬프트" and values:
            inputs["value"] = values[0]
        result[str(node["id"])] = {"inputs": inputs, "class_type": node["type"], "mode": node.get("mode", 0),
                                   "_meta": {"title": node.get("title", "")}}
    return result


def _basic_sampling_controls(nodes: dict) -> tuple[dict, dict]:
    """Locate the first sampler and only Sage patches on its upstream model."""
    defaults, bindings = {}, {}
    first = [(identity, node) for identity, node in nodes.items() if isinstance(node, dict)
             and node.get("class_type") == "SoyaFirstSamplerV2_mdsoya" and node.get("mode", 0) == 0
             and node.get("inputs", {}).get("sampling_key", "ANIMA_FIRST_SAMPLER") == "ANIMA_FIRST_SAMPLER"]
    for identity, node in first:
        mode = node.get("inputs", {}).get("sampler_mode")
        if isinstance(mode, str):
            if "sampler_mode" in defaults and defaults["sampler_mode"] != mode:
                raise ValueError("기본 생성 노드의 샘플러 종류가 서로 다릅니다.")
            defaults["sampler_mode"] = mode
            bindings.setdefault("sampler_mode", []).append((identity, "sampler_mode"))
        for name in FIRST_SAMPLER_NUMERIC_CONTROLS:
            value = node.get("inputs", {}).get(name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if name in defaults and defaults[name] != value:
                    raise ValueError(f"기본 생성 노드의 {name} 설정이 서로 다릅니다.")
                defaults[name] = value
                bindings.setdefault(name, []).append((identity, name))
    pending = [node.get("inputs", {}).get("model") for _, node in first]
    visited, sage = set(), []
    while pending:
        link = pending.pop()
        if not isinstance(link, list) or len(link) != 2 or str(link[0]) in visited:
            continue
        identity = str(link[0])
        visited.add(identity)
        node = nodes.get(identity, {})
        inputs = node.get("inputs", {})
        kind = node.get("class_type", "")
        if (kind == "PathchSageAttentionKJ" or kind.startswith("SoyaPathchSageAttentionKJV2_")) and node.get("mode", 0) == 0:
            mode = inputs.get("sage_attention")
            if isinstance(mode, str):
                sage.append(mode != "disabled")
                bindings.setdefault("sage_attention", []).append((identity, "sage_attention"))
        pending.extend(value for name, value in inputs.items() if "model" in name.lower())
    if sage:
        if len(set(sage)) != 1:
            raise ValueError("기본 생성 경로의 SAGE Attention 설정이 서로 다릅니다.")
        defaults["sage_attention"] = sage[0]
    return defaults, bindings


def _apply_basic_sampling_controls(workflow: dict, settings: dict) -> None:
    projected = workflow_json_settings_nodes(workflow)
    _, bindings = _basic_sampling_controls(projected)
    ui = isinstance(workflow.get("nodes"), list)
    nodes = {str(node["id"]): node for node in workflow["nodes"]} if ui else workflow
    for field, targets in bindings.items():
        for identity, name in targets:
            node = nodes[identity]
            value = settings[field]
            if field == "sage_attention":
                current = projected[identity]["inputs"][name]
                value = (current if current != "disabled" else "auto") if value else "disabled"
            if ui:
                widgets = [item for item in node["inputs"] if item.get("widget")]
                position = next(i for i, item in enumerate(widgets) if item["name"] == name)
                if widgets[position].get("link") is not None:
                    raise ValueError(f"{name}: 연결된 입력 대신 워크플로우 위젯을 사용하세요.")
                node["widgets_values"][position] = value
            else:
                node["inputs"][name] = value


def _normalize_prompt_keys(fields: dict) -> None:
    # Whole ANIMA prompts are assembled by the consuming node from their parts.
    fields.pop("ANIMA_ALL", None)
    if "SDXL" in fields:
        fields.setdefault("SDXL_CONTENT", fields.pop("SDXL"))


def illustration_fields(positive: str, negative: str = "") -> dict:
    """Normalize an editor draft or an already structured request once."""
    if positive.lstrip().startswith("{"):
        payload = json.loads(positive)
        if payload.get("schema") != SCHEMA:
            raise ValueError("Unsupported illustration JSON schema")
        if not isinstance(payload.get("fields"), dict):
            raise ValueError("Illustration JSON requires a fields object")
        _normalize_prompt_keys(payload["fields"])
        return copy.deepcopy(payload)
    fields = {}
    matches = list(_CONTROL_BLOCK_HEADER_RE.finditer(positive))
    for index, match in enumerate(matches):
        key = match.group("name")
        if key == "END":
            continue
        end = matches[index + 1].start() if index + 1 < len(matches) else len(positive)
        raw = positive[match.end():end].strip()
        if key in JSON_FIELDS:
            fields[key] = json.loads(raw) if raw else {}
        elif key in INT_FIELDS:
            fields[key] = int(raw)
        elif key in FLOAT_FIELDS:
            fields[key] = float(raw)
        elif key.endswith("_ACTIVATE") or key == "HRF_RESTORE_SIZE":
            fields[key] = raw.lower() == "true"
        elif key == "CHAR_LIST":
            fields[key] = [name.strip() for name in raw.split(",") if name.strip()]
        else:
            fields[key] = raw
    if not fields:
        raise ValueError("Illustration prompt has no structured fields")
    anima, marker, sdxl = negative.partition("[SDXL]")
    fields["NEGATIVE_ANIMA"] = anima.strip()
    fields["NEGATIVE_SDXL"] = sdxl.strip() if marker else anima.strip()
    # IPA is intentionally absent from the V2 contract. SnapRef uses images.
    fields.pop("FACE_ID_ACTIVATE", None)
    fields.pop("FACE_ID_STR", None)
    _normalize_prompt_keys(fields)
    return {"schema": SCHEMA, "fields": fields, "assets": {}}


def illustration_edit_document(positive: str, negative: str) -> dict:
    """Expose the existing editable prompt as one JSON document."""
    if positive.lstrip().startswith("{") or _CONTROL_BLOCK_HEADER_RE.search(positive):
        return illustration_fields(positive, negative)
    return {"positive": positive, "negative": negative}


def illustration_edit_prompts(document: dict, source_positive: str) -> tuple[str, str]:
    """Project JSON fields onto the existing scene editor and mask validators.

    The full document travels separately to the JSON workflow. Original path
    blocks remain available to the scene editor; embedded assets stay intact.
    """
    if not isinstance(document, dict):
        raise ValueError("프롬프트 JSON은 객체여야 합니다.")
    if "schema" not in document:
        if set(document) != {"positive", "negative"} or not all(isinstance(value, str) for value in document.values()):
            raise ValueError("프롬프트 JSON의 positive와 negative를 확인하세요.")
        return document["positive"], document["negative"]
    payload = illustration_fields(json.dumps(document, ensure_ascii=False))
    fields = payload["fields"]
    if not fields or not all(isinstance(key, str) and re.fullmatch(r"[A-Z][A-Z0-9_]*", key) for key in fields):
        raise ValueError("프롬프트 JSON의 fields를 확인하세요.")
    if not isinstance(payload.get("assets", {}), dict):
        raise ValueError("프롬프트 JSON의 assets는 객체여야 합니다.")
    for key in ("NEGATIVE_ANIMA", "NEGATIVE_SDXL"):
        if not isinstance(fields.get(key, ""), str):
            raise ValueError(f"{key}는 문자열이어야 합니다.")
    blocks = {}
    matches = list(_CONTROL_BLOCK_HEADER_RE.finditer(source_positive))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(source_positive)
        blocks[match.group("name")] = source_positive[match.end():end].strip()
    # These are the only draft-only fields omitted by the V2 JSON transport.
    preserved = {key: blocks[key] for key in ("CACHE_PATH", "FACE_ID_DIR", "FACE_ID_ACTIVATE", "FACE_ID_STR") if key in blocks}
    projected = {}
    for key, value in fields.items():
        if key in {"NEGATIVE_ANIMA", "NEGATIVE_SDXL", "FACE_EMBEDDINGS", "REGION_MASK"}:
            continue
        name = "SDXL" if key == "SDXL_CONTENT" else key
        if key == "CHAR_LIST":
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                raise ValueError("CHAR_LIST는 캐릭터 이름 배열이어야 합니다.")
            text = ", ".join(value)
        elif isinstance(value, str):
            text = value
        else:
            text = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        projected[name] = text
    if "ANIMA_CONTENT" in projected:
        projected["ANIMA_ALL"] = ", ".join(filter(None, [projected.get("ANIMA_QUALITY", ""), projected["ANIMA_CONTENT"]]))
    projected.update(preserved)
    positive = "\n".join(f"[{key}]\n{value}" for key, value in projected.items()) + "\n[END]"
    negative = fields.get("NEGATIVE_ANIMA", "") + "\n[SDXL]\n" + fields.get("NEGATIVE_SDXL", "")
    return positive, negative


def merge_illustration_edit_document(document: dict, before: tuple[str, str], after: tuple[str, str]) -> dict:
    """Apply only scene-editor changes, retaining sampling and embedded assets."""
    if document.get("schema") != SCHEMA:
        return illustration_edit_document(*after)
    result = copy.deepcopy(document)
    previous, updated = illustration_fields(*before)["fields"], illustration_fields(*after)["fields"]
    for key in previous.keys() | updated.keys():
        if previous.get(key) == updated.get(key):
            continue
        if key in updated:
            result["fields"][key] = updated[key]
        else:
            result["fields"].pop(key, None)
    return result


def _is_lora_data_key(key: str) -> bool:
    return key == "LORA_DATA" or key.endswith("_LORA_DATA")


_LORA_SET_SPECS: dict | None = None


def _illustration_v2_specs() -> dict:
    global _LORA_SET_SPECS
    if _LORA_SET_SPECS is None:
        path = (Path(__file__).parent / "comfy_installer" / "resources" / "illustration_v2"
                / "illustration_v2_specs.json")
        _LORA_SET_SPECS = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return _LORA_SET_SPECS


def _prompt_json_source(workflow: dict, identity: str, seen: frozenset[str] = frozenset()) -> str | None:
    """Follow pass-through prompt-JSON nodes (for example a render target selector)."""
    node = workflow.get(identity)
    if not isinstance(node, dict) or identity in seen:
        return None
    inputs = node.get("inputs", {})
    if isinstance(inputs.get("value"), str):
        return identity
    upstream = inputs.get("prompt_json")
    if isinstance(upstream, list) and len(upstream) == 2:
        return _prompt_json_source(workflow, str(upstream[0]), seen | {identity})
    return None


def _workflow_lora_data_sources(workflow: dict) -> dict[str, set[str]]:
    """LoRA set keys read by prompt-JSON loader nodes, and their source nodes."""
    specs = _illustration_v2_specs()
    result = {}
    for node in workflow.values():
        inputs = node.get("inputs", {}) if isinstance(node, dict) else {}
        prompt_json = inputs.get("prompt_json")
        if not isinstance(prompt_json, list) or len(prompt_json) != 2:
            continue
        spec = specs.get(str(node.get("class_type", "")))
        if not spec or spec.get("base") != "SoyaBatchLoraLoader_mdsoya":
            continue
        source = _prompt_json_source(workflow, str(prompt_json[0]))
        if source is None:
            continue
        for binding in spec.get("json_inputs", {}).values():
            key = str(binding.get("key", ""))
            if _is_lora_data_key(key):
                result.setdefault(key, set()).add(source)
    return result


def workflow_lora_data_keys(workflow: dict) -> list[str]:
    """LoRA sets edited as separate preset areas.

    STYLE_LORA_DATA stays a workflow-wide setting. Workflows with two or more
    branch loader sets (for example character and background VAEs) expose each
    branch set so every VAE can carry its own LoRAs.
    """
    sources = _workflow_lora_data_sources(workflow_json_settings_nodes(workflow))
    branch = [key for key in sources if key != "STYLE_LORA_DATA"]
    return ["STYLE_LORA_DATA", *branch] if len(branch) >= 2 else ["STYLE_LORA_DATA"]


def workflow_json_defaults(workflow: dict) -> dict:
    """Read selected sampling settings and model names from the workflow JSON."""
    workflow = workflow_json_settings_nodes(workflow)
    sources = {}
    for node in workflow.values():
        inputs = node.get("inputs", {}) if isinstance(node, dict) else {}
        prompt_json = inputs.get("prompt_json")
        if not isinstance(prompt_json, list) or len(prompt_json) != 2:
            continue
        selectors = [(selector, value_type) for selector, value_type in
                     (("sampling_key", dict), ("model_key", str)) if isinstance(inputs.get(selector), str)]
        if not selectors:
            continue
        source_id = _prompt_json_source(workflow, str(prompt_json[0]))
        if source_id is None:
            raise ValueError("Workflow JSON prompt source is not reachable")
        for selector, value_type in selectors:
            key = inputs[selector].strip().removeprefix("fields.")
            selected = sources.setdefault(source_id, {})
            if key in selected and selected[key] is not value_type:
                raise ValueError(f"Workflow JSON field {key} has conflicting uses")
            selected[key] = value_type
    lora_keys = workflow_lora_data_keys(workflow)
    lora_sources = _workflow_lora_data_sources(workflow)
    result = {}
    for source_id, keys in sources.items():
        source = workflow[source_id]["inputs"]["value"]
        payload = json.loads(source)
        for key, value_type in keys.items():
            settings = payload.get("fields", {}).get(key)
            if not isinstance(settings, value_type) or value_type is str and not settings.strip():
                raise ValueError(f"Workflow JSON prompt requires fields.{key}")
            if key in result and result[key] != settings:
                raise ValueError(f"Workflow JSON sources disagree on fields.{key}")
            result[key] = settings
        # Workflow-wide LoRAs are additive to the character cards' own LoRAs.
        for key in lora_keys:
            if key != "STYLE_LORA_DATA" and source_id not in lora_sources[key]:
                continue
            shared = payload.get("fields", {}).get(key, {"list": []})
            if key in result and result[key] != shared:
                raise ValueError(f"Workflow JSON sources disagree on {key}")
            result[key] = copy.deepcopy(shared)
        del payload
    controls, _ = _basic_sampling_controls(workflow)
    if isinstance(result.get("ANIMA_FIRST_SAMPLER"), dict):
        for name, value in controls.items():
            result["ANIMA_FIRST_SAMPLER"].setdefault(name, value)
    return result


def validate_workflow_json_settings(fields: dict, defaults: dict) -> None:
    """Validate exactly the fields consumed by this workflow's JSON nodes."""
    if not isinstance(fields, dict) or set(fields) != set(defaults):
        raise ValueError("워크플로우에서 사용하는 모델·샘플링·LoRA 설정을 모두 입력하세요.")
    for key, value in fields.items():
        if _is_lora_data_key(key):
            if not isinstance(value, dict) or set(value) != {"list"} or not isinstance(value["list"], list):
                raise ValueError("LoRA 설정에는 list 배열이 필요합니다.")
            for item in value["list"]:
                if (not isinstance(item, dict) or set(item) - {"lora_path", "str", "BASE", "trigger"}
                        or item.get("BASE") not in {"anima", "sdxl"}
                        or not isinstance(item.get("lora_path"), str) or not item["lora_path"].strip()
                        or not isinstance(item.get("str"), (int, float)) or isinstance(item["str"], bool)
                        or not math.isfinite(item["str"])
                        or not isinstance(item.get("trigger", ""), str)):
                    raise ValueError("LoRA 파일·계열·강도·트리거를 확인하세요.")
            continue
        if isinstance(defaults[key], str):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{key}: 모델 파일을 선택하세요.")
            continue
        if not isinstance(value, dict) or set(value) != set(defaults[key]):
            raise ValueError(f"{key}: 샘플링 설정 항목을 확인하세요.")
        for name, setting in value.items():
            if name == "sampler_mode" and setting not in FIRST_SAMPLER_MODES:
                raise ValueError(f"{key}.sampler_mode: 지원하는 샘플러 종류를 선택하세요.")
            if name in FIRST_SAMPLER_NUMERIC_CONTROLS:
                minimum, maximum = FIRST_SAMPLER_NUMERIC_CONTROLS[name]
                if (isinstance(setting, bool) or not isinstance(setting, (int, float))
                        or not math.isfinite(setting) or not minimum <= setting <= maximum):
                    raise ValueError(f"{key}.{name}: {minimum:g}~{maximum:g} 사이의 숫자를 입력하세요.")
            if isinstance(defaults[key][name], bool):
                if not isinstance(setting, bool):
                    raise ValueError(f"{key}.{name}: 켜기 또는 끄기를 선택하세요.")
            elif isinstance(defaults[key][name], str):
                if not isinstance(setting, str) or not setting.strip():
                    raise ValueError(f"{key}.{name}: 값을 입력하세요.")
            elif (isinstance(setting, bool) or not isinstance(setting, (int, float))
                  or not math.isfinite(setting) or setting < 0
                  or name == "steps" and (setting < 1 or int(setting) != setting)
                  or name == "denoise" and setting > 1):
                raise ValueError(f"{key}.{name}: 유효한 숫자를 입력하세요.")


def update_workflow_json_settings(workflow: dict, fields: dict) -> dict:
    defaults = workflow_json_defaults(workflow)
    if not defaults:
        raise ValueError("모델·샘플링 JSON 설정을 사용하는 삽화 워크플로우를 선택하세요.")
    fields = copy.deepcopy(fields)
    # Older presets inherit newly exposed controls from their source workflow.
    if isinstance(fields, dict) and isinstance(fields.get("ANIMA_FIRST_SAMPLER"), dict):
        for name in FIRST_SAMPLER_CONTROLS:
            if name in defaults.get("ANIMA_FIRST_SAMPLER", {}):
                fields["ANIMA_FIRST_SAMPLER"].setdefault(name, defaults["ANIMA_FIRST_SAMPLER"][name])
    if isinstance(fields, dict):
        for key, value in defaults.items():
            if _is_lora_data_key(key):
                fields.setdefault(key, copy.deepcopy(value))
    validate_workflow_json_settings(fields, defaults)
    result = copy.deepcopy(workflow)
    _apply_basic_sampling_controls(result, fields.get("ANIMA_FIRST_SAMPLER", {}))
    nodes = result["nodes"] if isinstance(result.get("nodes"), list) else result.values()
    updated = False
    for node in nodes:
        if node.get("title", node.get("_meta", {}).get("title")) != "긍정프롬프트":
            continue
        ui = "widgets_values" in node
        text = node["widgets_values"][0] if ui else node["inputs"]["value"]
        payload = illustration_fields(text)
        payload["fields"].update(copy.deepcopy(fields))
        serialized = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
        if ui:
            node["widgets_values"][0] = serialized
        else:
            node["inputs"]["value"] = serialized
        updated = True
    if not updated or workflow_json_defaults(result) != fields:
        raise ValueError("긍정프롬프트 JSON 설정과 연결된 워크플로우 노드를 확인하세요.")
    return result


def merge_workflow_loras(fields: dict, key: str, defaults: dict, *, include_triggers=True) -> None:
    """Merge one workflow LoRA set into a generation document.

    The request's own LoRAs (character cards, camera edits) stay first; the
    preset set is appended once per path and its triggers reach the prompt.
    """
    existing = fields.setdefault(key, {"list": []}).setdefault("list", [])
    for item in defaults.get("list", []):
        if not any(entry.get("BASE") == item["BASE"] and entry.get("lora_path", "").replace("\\", "/") == item["lora_path"].replace("\\", "/") for entry in existing):
            existing.append(copy.deepcopy(item))
        trigger = item.get("trigger", "").strip()
        if trigger and include_triggers:
            content = "ANIMA_CONTENT" if item["BASE"] == "anima" else "SDXL_CONTENT"
            fields[content] = ", ".join(filter(None, [trigger, fields.get(content, "")]))
            multi = fields.get("MULTI_CHAR")
            if item["BASE"] == "anima" and isinstance(multi, dict) and multi.get("enable"):
                # Regional sampling builds its own conditioning. Carry the
                # selected model preset's triggers there as well.
                if key == "LORA_DATA":
                    groups = multi.setdefault("char_trigger_list", [[] for _ in multi.get("char_name_list", [])])
                elif key == "BG_LORA_DATA":
                    groups = [multi.setdefault("background_trigger_list", [])]
                else:
                    groups = [multi.setdefault("shared_tag", {}).setdefault("before_char", [])]
                for group in groups:
                    if trigger not in group:
                        group.insert(0, trigger)
    fields[key.removesuffix("_DATA") + "_ACTIVATE"] = bool(existing)


def merge_shared_style_loras(fields: dict, defaults: dict) -> None:
    """Merge common model LoRAs and their triggers into a generation document."""
    merge_workflow_loras(fields, "STYLE_LORA_DATA", defaults)


def build_illustration_json(positive: str, negative: str, *, input_dir,
                            mask_location="region_mask", seed=None, workflow_defaults=None, bot_dir=None,
                            include_face_embeddings=True, render_target=None) -> str:
    payload = illustration_fields(positive, negative)
    fields = payload["fields"]
    if render_target is not None:
        if render_target not in {"character", "background"}:
            raise ValueError(f"잘못된 DUAL_VAE 출력 대상: {render_target}")
        fields["RENDER_TARGET"] = render_target
    for key, defaults in (workflow_defaults or {}).items():
        if _is_lora_data_key(key):
            if positive.lstrip().startswith("{"):
                # An already structured request is a complete saved generation.
                fields.setdefault(key, copy.deepcopy(defaults))
                fields.setdefault(key.removesuffix("_DATA") + "_ACTIVATE",
                                  bool((fields.get(key) or {}).get("list")))
                continue
            include_triggers = (render_target is None or key not in {"LORA_DATA", "BG_LORA_DATA"}
                                or key == ("LORA_DATA" if render_target == "character" else "BG_LORA_DATA"))
            merge_workflow_loras(fields, key, defaults, include_triggers=include_triggers)
            continue
        if key not in fields:
            fields[key] = copy.deepcopy(defaults)
        elif isinstance(defaults, dict) and isinstance(fields[key], dict):
            fields[key] = {**defaults, **fields[key]}
    if render_target is not None:
        # Both loaders are fed complete independent sets; the selector chooses
        # the model/VAE and suppresses a second sampler LoRA application.
        for key in ("LORA_DATA", "BG_LORA_DATA"):
            fields.setdefault(key, {"list": []})
            fields.setdefault(key.removesuffix("_DATA") + "_ACTIVATE", bool(fields[key]["list"]))
    assets = payload.setdefault("assets", {})
    for key in ("LORA_DATA", "FACE_LORA_DATA", "STYLE_LORA_DATA", "BG_LORA_DATA"):
        _normalize_payload_paths(fields.get(key, {}), path_fields=frozenset({"lora_path"}), block_name=key)
    def input_path(value):
        # Resolve the Comfy directory only for legacy path-based assets.
        # Original bot references and embedded JSON must not access it.
        root = Path(input_dir).resolve()
        path = (root / str(value).replace("\\", "/")).resolve()
        if not path.is_relative_to(root) or path == root:
            raise ValueError(f"Illustration input is outside INPUT: {value}")
        return path

    def embed(path):
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        assets.setdefault(digest, {"encoding": "base64", "name": path.name,
                                  "media_type": mimetypes.guess_type(path.name)[0] or "application/octet-stream",
                                  "sha256": digest, "data": base64.b64encode(data).decode("ascii")})
        return digest

    # Legacy face matching needs embeddings; spatial face matching does not.
    cache = fields.pop("CACHE_PATH", {})
    if not include_face_embeddings:
        # Spatial FD/ED matching reads geometry, including for a new NPC whose
        # cache has never existed. Saved JSON may still contain old embeddings.
        old_embeddings = fields.pop("FACE_EMBEDDINGS", [])
        image_assets = {entry.get(key) for entry in fields.get("SNAPREF", {}).get("references", [])
                        for key in ("asset", "face_asset", "source_asset", "face_source_asset")}
        image_assets.add(fields.get("REGION_MASK"))
        for entry in old_embeddings:
            if entry.get("asset") not in image_assets:
                assets.pop(entry.get("asset"), None)
    elif "FACE_EMBEDDINGS" not in fields:
        fields["FACE_EMBEDDINGS"] = [
            {"CHAR": item["CHAR"], "asset": embed(input_path(item["emb_path"]))}
            for item in cache.get("list", [])
        ]
    refs = fields.pop("FACE_ID_DIR", {})
    snapref = fields.setdefault("SNAPREF", {"enabled": False})
    if "references" not in snapref:
        snapref["references"] = []
        if snapref.get("enabled") and (snapref.get("strength_scale") != "linear" or float(snapref.get("strength", 1.0)) > 0):
            # A user-selected reference directory can override face identity data.
            override = snapref.pop("dir", None)
            entries = refs.get("list", [])
            if override:
                entries = [{"CHAR": (fields.get("CHAR_LIST") or [""])[0], "ipa_path": override}]
            bot_root = Path(bot_dir).resolve() if bot_dir else None
            for entry in entries:
                value = entry.get("snapref_path") or entry["ipa_path"]
                relative = Path(str(value).replace("\\", "/"))
                if ".." in relative.parts:
                    raise ValueError(f"Illustration input is outside INPUT: {value}")
                relative = relative.parent if relative.suffix else relative
                artifact = char_root = None
                if bot_root and relative.parts and relative.parts[0] == "soya_bot":
                    candidate = (bot_root / Path(*relative.parts[1:])).resolve()
                    if candidate.is_relative_to(bot_root):
                        artifact = candidate
                    if len(relative.parts) >= 3:
                        candidate = (bot_root / relative.parts[1] / relative.parts[2]).resolve()
                        if candidate.is_relative_to(bot_root):
                            char_root = candidate
                face = None
                if artifact and (artifact / "_face_image.webp").is_file():
                    face = (artifact / "_face_image.webp").resolve()
                images = []
                if char_root:
                    # The bot project owns the representative images; the input
                    # folder only holds staging copies for path-based workflows.
                    for name in entry.get("snapref_images") or []:
                        name = str(name).strip()
                        if not name or name in {".", ".."} or "/" in name or "\\" in name:
                            continue
                        candidate = (char_root / name).resolve()
                        if candidate.is_relative_to(char_root) and candidate.is_file():
                            images.append(candidate)
                if not images and artifact and char_root:
                    # Older prompts carry no image names; the card still records
                    # which representative its FACE was generated from.
                    record = artifact / "_face_image_source.json"
                    source = json.loads(record.read_text(encoding="utf-8")) if record.is_file() else {}
                    source = source.get("source_filename") if isinstance(source, dict) else ""
                    candidate = (char_root / str(source or "")).resolve()
                    if source and candidate.is_relative_to(char_root) and candidate.is_file():
                        images = [candidate]
                        print(f"[SnapRef] {entry['CHAR']}: 입력 폴더 대신 봇 데이터 참조: {source}")
                if not images and "snapref_images" not in entry:
                    # Old prompts without original filenames can still use their
                    # explicit input directory. New bot prompts never need it.
                    directory = input_path(relative)
                    staged = sorted(p for p in directory.glob("*") if p.is_file()
                                    and p.stem.casefold() == "representation"
                                    and p.suffix.lower() in IMAGE_EXTENSIONS)
                    if staged:
                        second = next((p for p in sorted(directory.glob("sub_1.*"))
                                       if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS), None)
                        images = [staged[0], *([second] if second else [])]
                if not images and face:
                    # Face-only cards still carry identity, though framing is tighter.
                    images = [face]
                    print(f"[SnapRef] {entry['CHAR']}: 대표 이미지가 없어 얼굴 이미지 사용")
                if images:
                    reference = {"CHAR": entry["CHAR"], "asset": embed(images[0])}
                    if face:
                        reference["face_asset"] = embed(face)
                    snapref["references"].append(reference)
                    secondary = snapref.get("secondary_reference", "none")
                    second = None
                    if secondary == "representative":
                        second = images[1] if len(images) > 1 else None
                    elif secondary == "face":
                        second = face if face != images[0] else None
                    if secondary in {"face", "representative"}:
                        if second is None:
                            label = "얼굴 이미지" if secondary == "face" else "두 번째 대표 이미지"
                            print(f"[SnapRef] {entry['CHAR']}: {label}가 없어 대표 이미지 1장 사용")
                            continue
                        snapref["references"].append({"CHAR": entry["CHAR"], "asset": embed(second),
                                                      "role": secondary})
    preprocess_reference_images(payload)
    multi = fields.get("MULTI_CHAR", {})
    if multi.get("enable") and "REGION_MASK" not in fields:
        directory = input_path(mask_location)
        images = sorted(p for p in directory.glob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        if not images:
            raise FileNotFoundError(f"Regional mask is missing: {directory}")
        fields["REGION_MASK"] = embed(images[0])
    if seed is not None:
        fields["SEED"] = int(seed)
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def is_illustration_json_workflow(workflow: dict) -> bool:
    return any(isinstance(node, dict) and str(node.get("class_type", "")).startswith("Soya")
               and "V2" in node.get("class_type", "") and "prompt_json" in node.get("inputs", {})
               for node in workflow.values())
