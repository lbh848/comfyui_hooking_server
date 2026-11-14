"""Asset model presets share the illustration editor's persistence contract."""
from __future__ import annotations

import copy
from pathlib import Path

from illustration_model_presets import IllustrationModelPresets
from illustration_json import merge_shared_style_loras
from preset_locations import PRESET_LOCATIONS

ASSET_WORKFLOW_PATHS = {
    "anima_only": "anima_only_asset_workflow_source_path",
    "anima_ilxl": "anima_asset_workflow_source_path",
}


class AssetModelPresets(IllustrationModelPresets):
    type_key = "asset_workflow_type"
    paths_key = "asset_workflow_source_paths"
    config_keys = (type_key, paths_key)
    local_profiles = tuple(ASSET_WORKFLOW_PATHS)
    local_profile = staticmethod(lambda profile: profile if profile in ASSET_WORKFLOW_PATHS else None)

    def __init__(self, config_file, builtin_file=None):
        super().__init__(config_file, builtin_file or PRESET_LOCATIONS["asset_model"].builtin_path())
        self.path = PRESET_LOCATIONS["asset_model"].local_path(config_file=config_file)
        # Assets keep a single preset list; never split it into workshop files.
        self.separates_workshops = False

    @staticmethod
    def selected_config(config):
        profile = config.get("asset_workflow_type", "anima_only")
        paths = config.get("asset_workflow_source_paths", {})
        return {"asset_workflow_type": profile if profile in ASSET_WORKFLOW_PATHS else "anima_only",
                "asset_workflow_source_paths": {profile: paths.get(profile, config.get(key, ""))
                                                for profile, key in ASSET_WORKFLOW_PATHS.items()}}

    def validate_config(self, candidate):
        paths = candidate.get(self.paths_key)
        if (candidate.get(self.type_key) not in self.local_profiles or not isinstance(paths, dict)
                or set(paths) - set(self.local_profiles) or not all(isinstance(path, str) for path in paths.values())):
            raise ValueError("ANIMA 또는 ANIMA+ILXL 그림 모델과 워크플로우를 선택하세요.")


def apply_asset_preset(document, preset):
    """Apply a captured preset, adding shared LoRAs to this asset's own LoRAs."""
    document = copy.deepcopy(document)
    profile = preset["config"]["asset_workflow_type"]
    settings = preset["workflows"][profile]["fields"]
    fields = document["fields"]
    for key, value in settings.items():
        if key != "STYLE_LORA_DATA":
            fields[key] = copy.deepcopy(value)
    if not fields.get("STYLE_LORA_ACTIVATE"):
        fields["STYLE_LORA_DATA"] = {"list": []}
    merge_shared_style_loras(fields, settings.get("STYLE_LORA_DATA", {}))
    return document
