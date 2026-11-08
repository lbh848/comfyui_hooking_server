"""Migrate converted illustration graphs to independent JSON-consuming V2 nodes.

Usage: python -m scripts.migrate_illustration_json_v2 SOURCE --output DEST
Requires the source workflow's nodes to be installed at --comfy-url.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import urllib.request

from illustration_json import illustration_fields


def migrate(workflow, api, info, specs):
    api = copy.deepcopy(api)
    source_nodes = {str(node["id"]): node for node in workflow["nodes"]}
    parser_id = next(key for key, node in api.items() if node["class_type"] == "SoyaPromptParser_mdsoya")
    positive_id = str(api[parser_id]["inputs"]["text"][0])
    negative_id = next(key for key, node in api.items() if node.get("_meta", {}).get("title") == "부정프롬프트")
    payload = illustration_fields(api[positive_id]["inputs"]["value"], api[negative_id]["inputs"]["value"])
    # Ship a runnable, neutral example without personal paths or example LoRAs.
    fields = payload["fields"]
    fields.update(CHAR_LIST=[], FACE_EMBEDDINGS=[], MULTI_CHAR={"enable": False}, SNAPREF={"enabled": False, "references": []})
    for key in ("CACHE_PATH", "FACE_ID_DIR"):
        fields.pop(key, None)
    for key in ("LORA", "FACE_LORA", "STYLE_LORA"):
        fields[key + "_ACTIVATE"] = False
        fields[key + "_DATA"] = {"list": []}
    fields["CHAR_FACE_TAG_INFORM"] = {"list": []}
    fields.update(ANIMA_QUALITY="masterpiece, best quality", ANIMA_ARTIST="",
                  ANIMA_CONTENT="1girl, standing in a garden",
                  SDXL_QUALITY="masterpiece, best quality", SDXL_ARTIST="", SDXL_CONTENT="masterpiece, best quality, 1girl, standing in a garden")
    bypass = {"LoraLoaderModelOnly": ("model",), "LoraLoader": ("model", "clip"),
              "Power Lora Loader (rgthree)": ("model", "clip"),
              "SoyaIPAApplyFacePatches_mdsoya": ("model",), "SoyaIPAdapterPatchCleaner_mdsoya": ("model",),
              "SoyaPassthrough_mdsoya": ("input",)}
    removed = {key for key, node in api.items() if node["class_type"] in bypass
               or key in {positive_id, negative_id}
               or node["class_type"] in {"SoyaPromptParser_mdsoya", "RegexExtract", "IPAdapterModelLoader", "CLIPVisionLoader"}}

    def resolve(value):
        if not isinstance(value, list) or len(value) != 2 or str(value[0]) not in api:
            return value
        source, slot = str(value[0]), value[1]
        node = api[source]
        kind = node["class_type"]
        if kind == "SoyaPromptParser_mdsoya":
            key = info[kind]["output_name"][slot]
            return {"json_key": key, "type": info[kind]["output"][slot]}
        if kind == "RegexExtract":
            key = "NEGATIVE_ANIMA" if "?=" in node["inputs"]["regex_pattern"] else "NEGATIVE_SDXL"
            return {"json_key": key, "type": "STRING"}
        if kind in bypass:
            return resolve(node["inputs"][bypass[kind][slot]])
        return [source, slot]

    for key, node in api.items():
        if key in removed:
            continue
        base = node["class_type"]
        bindings, inputs, fixed = {}, {}, {}
        for name, value in node["inputs"].items():
            value = resolve(value)
            if isinstance(value, dict) and "json_key" in value:
                bindings[name] = {"key": value["json_key"], "type": value["type"]}
            else:
                inputs[name] = value
        if base == "SoyaFirstSampler_mdsoya":
            fixed = {"mask_location": "", "FACE_ID_DIR": ""}
            bindings.pop("FACE_ID_DIR", None)
            for name in fixed:
                inputs.pop(name, None)
        # Turbo/DMD2 weights have been removed: retain ordinary sampling settings.
        if "sampler_name" in inputs and (inputs.get("sampler_name") == "lcm" or inputs.get("cfg", 5) < 2):
            inputs.update(sampler_name="er_sde", scheduler="simple", steps=max(20, inputs.get("steps", 20)), cfg=5.0)
        if base == "SoyaIPAPatchMaker_mdsoya":
            fixed = {"ipa_cache_data": "", "embed_cache_data": "", "combine_method": "average"}
            for name in fixed:
                inputs.pop(name, None)
                bindings.pop(name, None)
        signature = json.dumps([base, bindings, fixed], sort_keys=True)
        suffix = hashlib.sha256(signature.encode()).hexdigest()[:8]
        stem = base.removesuffix("_mdsoya")
        stem = {"SoyaIPAPatchMaker": "SoyaFaceMatch", "SoyaIPAPatchConfig": "SoyaFaceMatchConfig"}.get(stem, stem)
        stem = re.sub(r"[^A-Za-z0-9_]", "", stem)
        if not stem.startswith("Soya"):
            stem = "Soya" + stem
        name = stem + "V2_" + suffix + "_mdsoya"
        if base in {"SoyaFirstSampler_mdsoya", "SoyaIPAPatchMaker_mdsoya", "SoyaIPAPatchConfig_mdsoya"}:
            name = stem + "V2_mdsoya"
        if base == "CLIPTextEncode":
            # Keep the established V2 IDs while exposing their current key selection.
            json_key = bindings["text"]["key"]
            bindings["text"]["key"] = {"ANIMA_ALL": "ANIMA_QUALITY, ANIMA_CONTENT", "SDXL": "SDXL_CONTENT"}.get(json_key, json_key)
        schema = info[base]
        display = schema.get("display_name", base).replace("(Soya)", "").strip()
        display = {"SoyaIPAPatchMaker_mdsoya": "Face Match", "SoyaIPAPatchConfig_mdsoya": "Face Match Config"}.get(base, display)
        spec = {"name": name, "base": base, "display_name": display + " V2 (Soya JSON)",
                "json_inputs": bindings, "fixed_inputs": fixed,
                "output": schema["output"], "output_name": schema["output_name"],
                "output_is_list": schema["output_is_list"], "input_is_list": schema.get("is_input_list", False),
                "output_node": schema.get("output_node", False)}
        sampling = specs.get(name, {}).get("sampling")
        if sampling:
            spec["sampling"] = copy.deepcopy(sampling)
            declarations = {**schema["input"].get("required", {}), **schema["input"].get("optional", {})}
            settings = {}
            for field in sampling["inputs"]:
                declaration = declarations[field]
                default = (declaration[1].get("default") if len(declaration) > 1 else None)
                if default is None and isinstance(declaration[0], list):
                    default = declaration[0][0]
                value = inputs.pop(field, default)
                if isinstance(value, list):
                    raise ValueError(f"Sampling input {field} must be a value before JSON migration")
                settings[field] = value
            fields[sampling["key"]] = settings
            inputs["sampling_key"] = sampling["key"]
        model = specs.get(name, {}).get("model")
        if model:
            spec["model"] = copy.deepcopy(model)
            filename = inputs.pop(model["input"])
            if not isinstance(filename, str) or not filename.strip():
                raise ValueError(f"Model input {model['input']} must be a filename before JSON migration")
            fields[model["key"]] = filename
            inputs["model_key"] = model["key"]
        if name in specs and specs[name] != spec:
            raise ValueError(f"Conflicting V2 contract: {name}")
        specs[name] = spec
        inputs["prompt_json"] = [positive_id, 0]
        if base == "CLIPTextEncode":
            inputs["json_key"] = bindings["text"]["key"]
        node.update(class_type=name, inputs=inputs)
    api = {key: node for key, node in api.items() if key not in removed}
    api[positive_id] = {"class_type": "PrimitiveStringMultiline", "inputs": {"value": json.dumps(payload, ensure_ascii=False, indent=2)},
                        "_meta": {"title": "긍정프롬프트"}}
    # Removing LoRA/IPA branches can leave providers without a consumer.
    kept = set()
    def visit(key):
        if key in kept:
            return
        kept.add(key)
        for value in api[key]["inputs"].values():
            if isinstance(value, list) and len(value) == 2 and str(value[0]) in api:
                visit(str(value[0]))
    for key, node in api.items():
        if specs.get(node["class_type"], {}).get("output_node"):
            visit(key)
    api = {key: node for key, node in api.items() if key in kept}

    nodes, links = [], []
    for order, (key, node) in enumerate(api.items()):
        kind = node["class_type"]
        spec = specs.get(kind, {})
        # Installed V2 wrappers may retain a different widget order from their
        # legacy base (for example sampling_key precedes snapref_fidelity).
        schema = copy.deepcopy(info.get(kind, info[spec.get("base", kind)]))
        for group in ("required", "optional"):
            for field in list(schema["input"].get(group, {})):
                if (field in spec.get("json_inputs", {}) or field in spec.get("fixed_inputs", {})
                        or field in spec.get("sampling", {}).get("inputs", [])
                        or field == spec.get("model", {}).get("input")):
                    schema["input"][group].pop(field)
        if spec:
            schema["input"]["required"]["prompt_json"] = ["STRING", {"forceInput": True}]
            if spec["base"] == "CLIPTextEncode":
                schema["input"].setdefault("optional", {})["json_key"] = [
                    "STRING", {"default": spec["json_inputs"]["text"]["key"], "multiline": False}]
            if spec.get("sampling"):
                schema["input"].setdefault("optional", {})["sampling_key"] = [
                    "STRING", {"default": spec["sampling"]["key"], "multiline": False}]
            if spec.get("model"):
                schema["input"].setdefault("optional", {})["model_key"] = [
                    "STRING", {"default": spec["model"]["key"], "multiline": False}]
        original = source_nodes.get(key, {})
        ui = {"id": int(key), "type": kind, "pos": original.get("pos", [0, 0]), "size": [340, 220],
              "flags": {}, "order": order, "mode": 0, "inputs": [], "outputs": [], "properties": {"Node name for S&R": kind},
              "widgets_values": [], "title": node.get("_meta", {}).get("title", spec.get("display_name", kind))}
        declarations = {**schema["input"].get("required", {}), **schema["input"].get("optional", {})}
        for field, value in node["inputs"].items():
            if field not in declarations:
                declarations[field] = ["STRING", {"forceInput": True}]
        for field, declaration in declarations.items():
            dtype = declaration[0]
            options = declaration[1] if len(declaration) > 1 else {}
            value = node["inputs"].get(field)
            linked = isinstance(value, list) and len(value) == 2 and str(value[0]) in api
            widget = (isinstance(dtype, list) or dtype in {"STRING", "INT", "FLOAT", "BOOLEAN"}) and not options.get("forceInput")
            port = {"name": field, "type": "COMBO" if isinstance(dtype, list) else dtype, "link": None}
            if widget:
                port["widget"] = {"name": field}
                default = options.get("default", dtype[0] if isinstance(dtype, list) and dtype else ("" if dtype == "STRING" else 0))
                ui["widgets_values"].append(default if linked or value is None else value)
                if options.get("control_after_generate"):
                    ui["widgets_values"].append("fixed")
            if linked:
                link_id = len(links) + 1
                port["link"] = link_id
                links.append([link_id, int(value[0]), value[1], int(key), len(ui["inputs"]), port["type"]])
            ui["inputs"].append(port)
        for index, dtype in enumerate(schema["output"]):
            ui["outputs"].append({"name": schema["output_name"][index], "type": dtype, "links": [], "slot_index": index})
        nodes.append(ui)
    by_id = {node["id"]: node for node in nodes}
    for link in links:
        by_id[link[1]]["outputs"][link[2]]["links"].append(link[0])
    groups = [copy.deepcopy(group) for group in workflow.get("groups", []) if "IPA" not in group.get("title", "")]
    for group in groups:
        if group.get("title") == "인자분해구역":
            group["title"] = "JSON 프롬프트"
    result = {**workflow, "nodes": nodes, "links": links, "groups": groups, "last_node_id": max(by_id), "last_link_id": len(links),
              "extra": {**workflow.get("extra", {}), "soya_prompt_schema": "soya.illustration.v2"}}
    return result, api


def migrate_settings(workflow, api, specs, *, kind):
    """Move current V2 sampling or model widgets into selected JSON fields."""
    if kind not in {"sampling", "model"}:
        raise ValueError(f"Unsupported JSON settings kind: {kind}")
    selector = kind + "_key"
    result = copy.deepcopy(workflow)
    by_id = {str(node["id"]): node for node in result["nodes"]}
    payloads = {}
    for node in result["nodes"]:
        binding = specs.get(node["type"], {}).get(kind)
        if not binding or any(port["name"] == selector for port in node.get("inputs", [])):
            continue
        field_names = binding["inputs"] if kind == "sampling" else [binding["input"]]
        inputs = api[str(node["id"])]["inputs"]
        source_id = str(inputs["prompt_json"][0])
        if source_id not in payloads:
            source = by_id[source_id]
            if source["type"] != "PrimitiveStringMultiline":
                raise ValueError("JSON settings require a JSON string source")
            payloads[source_id] = json.loads(source["widgets_values"][0])
        payload = payloads[source_id]
        if payload.get("schema") != "soya.illustration.v2":
            raise ValueError("JSON settings require a soya.illustration.v2 prompt")
        widget_names = [port["name"] for port in node["inputs"] if port.get("widget")]
        values = node.get("widgets_values", [])
        if len(widget_names) != len(values):
            raise ValueError(f'Cannot match widget values on node {node["id"]}')
        widgets = dict(zip(widget_names, values))
        settings = {}
        for name in field_names:
            port = next(port for port in node["inputs"] if port["name"] == name)
            if port.get("link") is not None or widgets[name] != inputs[name]:
                raise ValueError(f'Inspect linked or mismatched input {node["id"]}.{name}')
            settings[name] = inputs[name]
        key = binding["key"]
        if kind == "sampling":
            existing = payload["fields"].get(key, {})
            if not isinstance(existing, dict):
                raise ValueError(f"fields.{key} must be an object")
            payload["fields"][key] = {**settings, **existing}
        else:
            filename = payload["fields"].get(key, settings[binding["input"]])
            if not isinstance(filename, str) or not filename.strip():
                raise ValueError(f"fields.{key} must be a model filename")
            payload["fields"][key] = filename
        slots = {}
        kept = []
        for index, port in enumerate(node["inputs"]):
            if port["name"] not in field_names:
                slots[index] = len(kept)
                kept.append(port)
        for link in result["links"]:
            if link[3] == node["id"]:
                link[4] = slots[link[4]]
        kept.append({"name": selector, "type": "STRING", "widget": {"name": selector}, "link": None})
        node["inputs"] = kept
        node["widgets_values"] = [value for name, value in zip(widget_names, values)
                                   if name not in field_names] + [key]
    for source_id, payload in payloads.items():
        by_id[source_id]["widgets_values"][0] = json.dumps(payload, ensure_ascii=False, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--comfy-url", default="http://127.0.0.1:8188")
    settings = parser.add_mutually_exclusive_group()
    settings.add_argument("--sampling-settings", action="store_true",
                          help="Move existing V2 sampling widgets into JSON using the currently loaded node definitions")
    settings.add_argument("--model-settings", action="store_true",
                          help="Move existing V2 loader model names into JSON using the currently loaded node definitions")
    args = parser.parse_args()
    workflow = json.loads(args.source.read_text(encoding="utf-8"))
    info = json.load(urllib.request.urlopen(args.comfy_url + "/object_info"))
    request = urllib.request.Request(args.comfy_url + "/workflow/convert", data=json.dumps(workflow).encode(), headers={"Content-Type": "application/json"})
    api = json.load(urllib.request.urlopen(request))
    spec_path = Path(__file__).resolve().parents[1] / "comfy_installer/resources/illustration_v2/illustration_v2_specs.json"
    specs = json.loads(spec_path.read_text(encoding="utf-8")) if spec_path.exists() else {}
    if args.sampling_settings or args.model_settings:
        result = migrate_settings(workflow, api, specs, kind="model" if args.model_settings else "sampling")
    else:
        result, _ = migrate(workflow, api, info, specs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    spec_path.write_text(json.dumps(specs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Migrated {args.output.name}: {len(result['nodes'])} nodes")


if __name__ == "__main__":
    main()
