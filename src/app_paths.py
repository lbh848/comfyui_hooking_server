"""Installation layout shared by the existing engine and future application UIs.

Source and bundled defaults live in src; user settings and writable content live
beside it in config and data. A project_root argument denotes the source folder.
"""
from __future__ import annotations

from pathlib import Path
import copy
import shutil


SOURCE_DIR = Path(__file__).resolve().parent
INSTALL_DIR = SOURCE_DIR.parent
CONFIG_DIR = INSTALL_DIR / "config"
DATA_DIR = INSTALL_DIR / "data"
WORKFLOW_DIR = "data/comfy/user/default/workflows"
WORKFLOW_FILENAMES = {
    "comfy_workflow_source_path": "배포_공방_삽화_ANIMA_V4.json",
    "illustration_workflow_source_paths.v3": "배포_공방_삽화_ANIMA+ILXL_V4.json",
    "illustration_workflow_source_paths.v3_anima": "배포_공방_삽화_ANIMA_V4.json",
    "utility_workflow_source_path": "배포_공방_삽화유틸리티_V2_0.json",
    "anima_asset_workflow_source_path": "배포_공방_에셋_ANIMA+ILXL_V4.json",
    "anima_only_asset_workflow_source_path": "배포_공방_에셋_ANIMA_V4.json",
    "moss_asset_workflow_source_path": "배포_공방_에셋_MOSS_8STEP_V1.json",
    "anima_inpainting_workflow_source_path": "배포_공방_인페인트_ANIMA_V2_0.json",
    "tag_analysis_workflow_source_path": "배포_공방_범용태그분석_ANIMA_V1_0.json",
    "face_extract_workflow_source_path": "배포_공방_얼굴추출_V1_0.json",
    "debug_workflow_source_path": "배포_공방_디버깅_V1_0.json",
    "gemma4_workflow_source_path": "배포_공방_Gemma4_이미지분석_V1_0.json",
    "lora_training_workflow_source_paths.anima": "배포_공방_LoRA학습_캐릭터_ANIMA_V2_0.json",
    "lora_training_workflow_source_paths.sdxl": "배포_공방_LoRA학습_캐릭터_ILXL_V1_1.json",
    "lora_training_workflow_source_paths.moss": "배포_공방_LoRA학습_캐릭터_MOSS_V2_0.json",
    "style_lora_training_workflow_source_paths.anima": "배포_공방_LoRA학습_스타일_ANIMA_V2_0.json",
    "style_lora_training_workflow_source_paths.sdxl": "배포_공방_LoRA학습_스타일_ILXL_V1_1.json",
    "style_lora_training_workflow_source_paths.moss": "배포_공방_LoRA학습_스타일_MOSS_V2_0.json",
    "workflow_pack_extra_paths.anima_rgba_asset": "배포_공방_ANIMA_V4_투명배경.json",
    "workflow_pack_extra_paths.illustration_dual_vae": "배포_공방_삽화_DUAL_VAE_V1.json",
    "workflow_pack_extra_paths.illustration_moss_4step": "배포_공방_삽화_MOSS_4STEP_V1.json",
    "workflow_pack_extra_paths.illustration_moss_8step": "배포_공방_삽화_MOSS_8STEP_V1.json",
    "workflow_pack_extra_paths.asset_moss_4step": "배포_공방_에셋_MOSS_4STEP_V1.json",
    "workflow_pack_extra_paths.asset_moss_8step": "배포_공방_에셋_MOSS_8STEP_V1.json",
    "workflow_pack_extra_paths.asset_moss_50step": "배포_공방_에셋_MOSS_50STEP_V1.json",
    "workflow_pack_extra_paths.lora_training_anima_v2": "배포_공방_LoRA학습_캐릭터_ANIMA_V2_0.json",
    "workflow_pack_extra_paths.lora_training_anima_v2_1": "배포_공방_LoRA학습_캐릭터_ANIMA_V2_1.json",
    "workflow_pack_extra_paths.style_lora_training_anima_v2": "배포_공방_LoRA학습_스타일_ANIMA_V2_0.json",
    "workflow_pack_extra_paths.lora_training_moss_v2": "배포_공방_LoRA학습_캐릭터_MOSS_V2_0.json",
    "workflow_pack_extra_paths.style_lora_training_moss_v2": "배포_공방_LoRA학습_스타일_MOSS_V2_0.json",
}
TAG_ANALYSIS_WORKFLOW_PATHS = {
    key: f"{WORKFLOW_DIR}/{WORKFLOW_FILENAMES[key]}"
    for key in ("tag_analysis_workflow_source_path",)
}


def portable_tag_analysis_workflow_paths(config: dict) -> dict[str, str]:
    """Keep the distribution's tagger bindings fixed and portable."""
    return {key: value for key, value in TAG_ANALYSIS_WORKFLOW_PATHS.items() if key in config}


def install_root(project_root: str | Path = SOURCE_DIR) -> Path:
    root = Path(project_root).resolve()
    return root.parent if root.name == "src" else root


def data_root(project_root: str | Path = SOURCE_DIR) -> Path:
    return install_root(project_root) / "data"


def comfy_input_root(project_root: str | Path = SOURCE_DIR) -> Path:
    """The managed Comfy installation always owns this input directory."""
    return data_root(project_root) / "comfy" / "input"


def config_root(project_root: str | Path = SOURCE_DIR) -> Path:
    return install_root(project_root) / "config"


def resolve_install_path(value: str, project_root: str | Path = SOURCE_DIR) -> Path:
    """Resolve portable resource paths independently of the working directory."""
    path = Path(value.replace("\\", "/"))
    return path if path.is_absolute() else install_root(project_root) / path


def default_app_paths() -> dict:
    """Portable defaults for supported bundled workflows and Comfy folders."""
    defaults = {"workflow_base_dir": WORKFLOW_DIR, "comfy_input_dir": "data/comfy/input"}
    for binding, filename in WORKFLOW_FILENAMES.items():
        key, separator, profile = binding.partition(".")
        value = f"{WORKFLOW_DIR}/{filename}"
        if separator:
            defaults.setdefault(key, {})[profile] = value
        else:
            defaults[key] = value
    return defaults


def portable_app_paths(config: dict, project_root: str | Path = SOURCE_DIR, *, fill_defaults: bool = False) -> dict:
    """Fill supported paths and store in-installation selections relatively."""
    defaults = default_app_paths()
    root = install_root(project_root)

    def portable(value, default):
        if not isinstance(value, str) or not value.strip():
            return default
        path = Path(value.replace("\\", "/"))
        if path.is_absolute():
            try:
                return path.resolve().relative_to(root).as_posix()
            except ValueError:
                return value
        return path.as_posix()

    result = {}
    for key, default in defaults.items():
        if key not in config and not fill_defaults:
            continue
        if isinstance(default, dict):
            stored = config.get(key)
            paths = copy.deepcopy(stored) if isinstance(stored, dict) else {}
            profiles = {**dict.fromkeys(paths, ""), **default}
            result[key] = {profile: portable(paths.get(profile), path) for profile, path in profiles.items()}
        else:
            result[key] = portable(config.get(key), default)
    result.update(portable_tag_analysis_workflow_paths({**config, **result}))
    return result


def resolve_app_paths(config: dict, project_root: str | Path = SOURCE_DIR) -> dict:
    """Resolve managed data paths for existing runtime and installer callers."""
    def resolve(value):
        if isinstance(value, str) and value.replace("\\", "/").startswith("data/"):
            return str(resolve_install_path(value, project_root))
        return value

    defaults = default_app_paths()
    return {key: {profile: resolve(path) for profile, path in value.items()} if isinstance(value, dict) else resolve(value)
            for key, value in config.items() if key in defaults}


def initialize_layout(project_root: str | Path = SOURCE_DIR) -> None:
    """Create writable roots and seed missing defaults without replacing edits."""
    source = Path(project_root).resolve()
    config_root(source).mkdir(parents=True, exist_ok=True)
    destination = data_root(source)
    destination.mkdir(parents=True, exist_ok=True)
    for bundled_name, data_name in (
        ("prompts", "prompts"),
        ("auto_complete", "auto_complete"),
        ("defaults/illustration", "asset_data"),
        ("defaults/fonts", "fonts"),
    ):
        bundled = source / bundled_name
        if not bundled.is_dir():
            continue
        for original in bundled.rglob("*"):
            if not original.is_file() or "__pycache__" in original.parts:
                continue
            target = destination / data_name / original.relative_to(bundled)
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                with target.open("xb") as output, original.open("rb") as input_file:
                    shutil.copyfileobj(input_file, output)
            except FileExistsError:
                pass
    from data_backups import backup_presets
    backup_presets(source)


if __name__ == "__main__":
    initialize_layout()
