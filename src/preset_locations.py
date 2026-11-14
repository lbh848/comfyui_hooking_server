"""Preset storage addresses only; editing and unlock rules stay in each store.

Paths are relative to the installation. Directory entries contain per-preset
JSON files. This registry never copies, restores, locks, or validates content.
"""
from dataclasses import dataclass
from pathlib import Path

from app_paths import SOURCE_DIR, install_root


@dataclass(frozen=True)
class PresetLocation:
    builtin: str
    local: str
    directory: bool = False

    def builtin_path(self, project_root: str | Path = SOURCE_DIR) -> Path:
        return Path(project_root) / Path(self.builtin).relative_to("src")

    def local_path(self, project_root: str | Path = SOURCE_DIR, *, config_file: str | Path | None = None) -> Path:
        if config_file is not None and self.local.startswith("config/"):
            return Path(config_file).with_name(Path(self.local).name)
        return install_root(project_root) / self.local


PRESET_LOCATIONS = {
    "tags": PresetLocation("src/presets/presets.json", "data/asset_data/tags.json"),
    "chains": PresetLocation("src/presets/chain_presets", "data/chain_presets", directory=True),
    "asset_character": PresetLocation("src/presets/asset_character_presets.json", "data/asset_data/tags.json"),
    "asset_style": PresetLocation("src/presets/asset_style_presets.json", "config/asset_style_presets.json"),
    "asset_model": PresetLocation("src/presets/asset_model_presets.json", "config/asset_model_presets.json"),
    "illustration_model": PresetLocation("src/presets/illustration_model_presets.json", "config/illustration_model_presets.json"),
    "emotion_model": PresetLocation("src/presets/emotion_model_presets.json", "config/emotion_model_presets.json"),
    "illustration_generation": PresetLocation("src/presets/illustration_generation_presets.json", "config/illustration_generation_presets.json"),
    "emotion_generation": PresetLocation("src/presets/emotion_generation_presets.json", "config/emotion_generation_presets.json"),
    "illustration_bot": PresetLocation("src/defaults/illustration/bot.json", "data/asset_data/bot.json"),
    "emotion_bot": PresetLocation("src/defaults/emotion/bot.json", "data/asset_data/emotion_bot.json"),
    "bot_system_prompt": PresetLocation("src/prompts/bot_system_prompt/presets.json", "data/asset_data/bot.json"),
    "emotion_groups": PresetLocation("src/presets/emotion_preset_groups.json", "data/asset_data/emotion_preset_groups.json"),
    "emotion_background": PresetLocation("src/presets/emotion_background_presets.json", "config/emotion_background_presets.json"),
    "emotion_background_bot": PresetLocation("src/presets/emotion_background_bots.json", "config/emotion_background_bots.json"),
    "lora_training": PresetLocation("src/presets/lora_training_presets.json", "config/lora_training_presets.json"),
}

# Workflow ownership is determined by the encrypted pack, not preset scopes.
WORKFLOW_LOCATIONS = {
    "pack": "src/comfy_installer/resources/workflows.soyawfp",
    "distribution": "data/comfy_workflow_library/SOYA_DISTRIBUTION",
    "installed": "data/comfy/user/default/workflows",
    "groups": "config/workflow-groups.json",
}
