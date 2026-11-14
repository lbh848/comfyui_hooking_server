"""Resolve current distribution preset requirements without changing the pack."""
from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

from asset_model_presets import ASSET_WORKFLOW_PATHS, AssetModelPresets
from illustration_json import enabled_lora_list, workflow_json_defaults
from illustration_model_presets import IllustrationModelPresets
from preset_locations import PRESET_LOCATIONS

from .manifest import InstallManifest


def distribution_preset_models(manifest: InstallManifest, items: list[dict], *, workflow_root: Path | None = None,
                               include_user: bool = False) -> dict:
    """Only each shipped preset's selected local profile contributes requirements.

    User catalogs and inactive editor snapshots do not define distribution
    downloads. Re-read shipped files so edits made after unlocking take effect.
    Unknown files are reported per workflow; editing presets stays unrestricted.
    Cleanup includes user presets too, to protect all installed consumers.
    """
    source = manifest.source_path.parent
    if source.name == "resources" and source.parent.name == "comfy_installer":
        source = source.parent.parent
    by_filename = {}
    for item in items:
        by_filename.setdefault(PurePosixPath(str(item.get("filename", "")).replace("\\", "/")).name.casefold(), []).append(item)
    by_binding = {binding: item for item in items for binding in item.get("bindings", [])}
    result = {item["id"]: {"ids": set(), "errors": []} for item in items}
    model_paths = {}
    for model in manifest.models:
        path = str(model["relative_path"]).replace("\\", "/").casefold().removeprefix("models/")
        model_paths.setdefault(path, set()).add(model["id"])

    def resolve(value: str, folders: tuple[str, ...]) -> set[str]:
        path = value.replace("\\", "/").casefold().removeprefix("models/")
        if path.startswith(folders):
            return model_paths.get(path, set())
        return set().union(*(model_paths.get(folder + path, set()) for folder in folders))

    for key, store in (("illustration_model", IllustrationModelPresets),
                       ("emotion_model", IllustrationModelPresets), ("asset_model", AssetModelPresets)):
        path = PRESET_LOCATIONS[key].builtin_path(source)
        documents = [json.loads(path.read_text(encoding="utf-8-sig"))] if path.is_file() else []
        local = PRESET_LOCATIONS[key].local_path(source)
        if include_user and local.is_file():
            documents.append(json.loads(local.read_text(encoding="utf-8-sig")))
        for preset in (preset for document in documents for preset in document.get("presets", [])):
            config = preset.get("config", {})
            profile = store.local_profile(config.get(store.type_key))
            if not profile:
                continue
            workflow = str(config.get(store.paths_key, {}).get(profile, ""))
            if workflow.startswith("@configured:"):
                configured = workflow.removeprefix("@configured:")
                binding = (ASSET_WORKFLOW_PATHS.get(configured, "") if store is AssetModelPresets
                           else f"illustration_workflow_source_paths.{configured}")
                matches = [by_binding[binding]] if binding in by_binding else []
            else:
                matches = by_filename.get(PurePosixPath(workflow.replace("\\", "/")).name.casefold(), [])
            if not matches:
                continue
            fields = preset.get("workflows", {}).get(profile, {}).get("fields")
            if fields is None and workflow_root is not None:
                fields = workflow_json_defaults(json.loads((workflow_root / matches[0]["filename"]).read_text(encoding="utf-8-sig")))
            fields = fields or {}
            references = []
            for field, folders in (
                ("ANIMA_DIFFUSION_MODEL", ("diffusion_models/",)),
                ("ILXL_CHECKPOINT", ("checkpoints/",)),
                ("workflow.model", ("diffusion_models/", "checkpoints/")),
                ("workflow.clip", ("text_encoders/", "clip/")),
                ("workflow.vae", ("vae/",)),
            ):
                if fields.get(field):
                    references.append((str(fields[field]), folders))
            for field, value in fields.items():
                if field == "LORA_DATA" or field.endswith("_LORA_DATA"):
                    references.extend((str(lora["lora_path"]), ("loras/",))
                                      for lora in enabled_lora_list(value) if lora.get("lora_path"))
            for name, folders in references:
                identities = resolve(name, folders)
                for item in matches:
                    requirements = result[item["id"]]
                    if len(identities) == 1:
                        requirements["ids"].update(identities)
                    else:
                        requirements["errors"].append(
                            f"배포 프리셋 ‘{preset.get('name', key)}’의 모델을 설치 명세에서 확인하세요: {name}")
    return result
