"""Migrate ANIMA asset graphs using the illustration JSON nodes and routing."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import urllib.request

from asset_json import asset_document
from scripts.migrate_illustration_json_v2 import migrate
from scripts.organize_illustration_prompt_routes import organize_model_routes, organize_prompt_routes


def migrate_asset(workflow, api, info, specs):
    workflow, api, info = copy.deepcopy(workflow), copy.deepcopy(api), copy.deepcopy(info)
    parser_id = next(key for key, node in api.items() if node["class_type"] == "SoyaAssetV2PromptParser_mdsoya")
    parser = api[parser_id]
    schema = copy.deepcopy(info[parser["class_type"]])
    schema["output_name"] = [{"ANIMA": "ANIMA_ALL"}.get(key, key) for key in schema["output_name"]]
    schema["output_name"].append("MULTI_CHAR")
    schema["output"].append("STRING")
    info["SoyaPromptParser_mdsoya"] = schema
    parser["class_type"] = "SoyaPromptParser_mdsoya"
    positive_id = str(parser["inputs"]["text"][0])
    negative = next(node["inputs"]["value"] for node in api.values() if node.get("_meta", {}).get("title") == "부정프롬프트")
    payload = asset_document(api[positive_id]["inputs"]["value"], negative)
    api[positive_id]["inputs"]["value"] = json.dumps(payload, ensure_ascii=False)

    # The shared first sampler applies character LoRAs itself. Keep the legacy
    # batch loader for hires/hands, but remove it from the first sampling path.
    def sampler_model(value):
        if not isinstance(value, list) or len(value) != 2:
            return value
        identity = str(value[0])
        source = api[identity]
        inputs = source["inputs"]
        if source["class_type"] == "SoyaBatchLoraLoader_mdsoya" and inputs.get("base_filter") == "anima":
            enable = inputs.get("enable")
            if enable == [parser_id, schema["output_name"].index("LORA_ACTIVATE")]:
                return copy.deepcopy(inputs["model"])
        if "model" in inputs:
            replacement = sampler_model(inputs["model"])
            if replacement != inputs["model"]:
                consumers = sum(value == [identity, 0] for node in api.values() for value in node["inputs"].values())
                if consumers != 1:
                    raise ValueError("The first sampler shares its pre-LoRA model patch with another stage")
                inputs["model"] = replacement
        return value

    for node in api.values():
        if node["class_type"] == "SoyaFirstSampler_mdsoya":
            node["inputs"]["model"] = sampler_model(node["inputs"]["model"])

    # Reconnect conditioning around ControlNet before discarding its providers.
    for node in api.values():
        for name, value in list(node["inputs"].items()):
            if isinstance(value, list) and len(value) == 2 and str(value[0]) in api:
                source = api[str(value[0])]
                if source["class_type"] == "SoyaControlNetToggle_mdsoya":
                    if value[1] not in (0, 1):
                        raise ValueError("Unexpected ControlNet diagnostic consumer")
                    node["inputs"][name] = copy.deepcopy(source["inputs"][("positive", "negative")[value[1]]])
        if node["class_type"] == "SoyaFirstSampler_mdsoya":
            for name, binding in specs["SoyaFirstSamplerV2_mdsoya"]["json_inputs"].items():
                node["inputs"][name] = [parser_id, schema["output_name"].index(binding["key"])]
            node["inputs"]["snapref_model"] = info["SoyaFirstSamplerV2_mdsoya"]["input"]["optional"]["snapref_model"][1]["default"]
        if node["class_type"] == "SoyaIPAPatchMaker_mdsoya":
            node["inputs"].update(ipa_cache_data="", embed_cache_data="")

    removed = {key for key, node in api.items() if any(token in node["class_type"] for token in (
        "ControlNet", "IPAdapter", "InsightFace", "EmbedCache", "CombineEmbeds", "WeightCalc", "CLIPVision"))}
    # Patch cleaners only pass the model through; bypass before pruning.
    def resolve(value):
        if isinstance(value, list) and len(value) == 2 and str(value[0]) in api:
            source = api[str(value[0])]
            if source["class_type"] in {"SoyaIPAdapterPatchCleaner_mdsoya", "SoyaIPAdapterEmbedsToggle_mdsoya"}:
                return resolve(source["inputs"]["model"])
        return value
    for node in api.values():
        node["inputs"] = {key: resolve(value) for key, value in node["inputs"].items()}
    api = {key: node for key, node in api.items() if key not in removed}
    # Dead preview/debug branches must not retain removed provider inputs.
    while True:
        invalid = {key for key, node in api.items() if any(isinstance(value, list) and len(value) == 2
                   and str(value[0]) in removed for value in node["inputs"].values())}
        if not invalid:
            break
        removed.update(invalid)
        api = {key: node for key, node in api.items() if key not in invalid}
    workflow["groups"] = [group for group in workflow.get("groups", [])
                          if "IPA" not in group.get("title", "")]
    result, _ = migrate(workflow, api, info, specs)
    prompt = next(node for node in result["nodes"] if node.get("title") == "긍정프롬프트")
    document = json.loads(prompt["widgets_values"][0])
    fields = document["fields"]
    fields.update(CHAR_LIST=["asset_mode"], IMG_W=700, IMG_H=1024, SEED=0,
                  ANIMA_CONTENT="1girl, standing, simple background",
                  SDXL_CONTENT="masterpiece, best quality, 1girl, standing, simple background")
    for key in list(fields):
        if key.endswith("_ACTIVATE"):
            fields[key] = False
    document = asset_document(json.dumps(document))
    prompt["widgets_values"][0] = json.dumps(document, ensure_ascii=False, indent=2)
    result = organize_prompt_routes(result)
    result = organize_model_routes(result)
    if not any(node["type"].startswith("SoyaPreviewImageV2") for node in result["nodes"]):
        raise ValueError("Asset migration lost the image output")
    result.setdefault("extra", {})["soya_asset_json"] = 1
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--comfy-url", default="http://127.0.0.1:8188")
    args = parser.parse_args()
    workflow = json.loads(args.source.read_text(encoding="utf-8"))
    info = json.load(urllib.request.urlopen(args.comfy_url + "/object_info", timeout=30))
    request = urllib.request.Request(args.comfy_url + "/workflow/convert", data=json.dumps(workflow).encode(), headers={"Content-Type": "application/json"})
    api = json.load(urllib.request.urlopen(request, timeout=30))
    spec_path = Path(__file__).resolve().parents[1] / "comfy_installer/resources/illustration_v2/illustration_v2_specs.json"
    specs = json.loads(spec_path.read_text(encoding="utf-8"))
    result = migrate_asset(workflow, api, info, specs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    spec_path.write_text(json.dumps(specs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Migrated {args.output.name}: {len(result['nodes'])} nodes")


if __name__ == "__main__":
    main()
