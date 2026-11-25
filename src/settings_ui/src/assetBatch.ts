import { assetModelPreset, assetPresetFields, copyAssetCharacterSettings, copyAssetStyleSettings, newAssetSlot, type AssetRecord, type AssetSettingsSection } from "./asset";

export const assetBatchSkip = "__asset_batch_skip__";

export function assetBatchSlot(slot: AssetRecord, selection: AssetRecord | null): AssetRecord {
  // Saved chains and styles may contain an old model ID. All batch slots follow
  // the active model, including when a different model is selected after loading.
  return { ...slot, asset_model_preset_id: "",
    asset_workflow_type: assetModelPreset({}, selection)?.config.asset_workflow_type || slot.asset_workflow_type };
}

export function assetBatchSelectValue(patch: AssetRecord | undefined, keys: string[], value: string): string {
  return patch && !keys.some(key => patch[key] !== undefined) ? assetBatchSkip : value;
}

export function updateAssetBatchPatch(previous: AssetRecord, changes: AssetRecord): AssetRecord {
  // Undefined removes an override; empty strings, false and zero are actual changes.
  return Object.fromEntries(Object.entries({ ...previous, ...changes }).filter(([, value]) => value !== undefined));
}

export function assetBatchSettingsValues(patch: AssetRecord): AssetRecord {
  const values = { ...newAssetSlot(), ...patch };
  // The displayed form values are also the values copied by Apply to all slots.
  // An unchecked checkbox must overwrite a previously loaded true value.
  for (const key of ["snapref_enabled", "hrf_anima", "hrf_sdxl", "hrf_moss", "hrf_restore_size",
    "anima_fd_enabled", "anima_hd_enabled", "anima_ed_enabled", "sdxl_fd_enabled", "sdxl_hd_enabled", "sdxl_ed_enabled",
    "lora_enabled", "face_lora_enabled", "style_lora_enabled"]) values[key] = !!values[key];
  delete values._id;
  // Preset selectors retain their explicit skip behavior and slot identities.
  for (const key of ["character", "asset_workflow_type", ...assetPresetFields.map(([key]) => key)]) {
    if (patch[key] === undefined) delete values[key];
  }
  return values;
}

export function excludeAssetBatchSection(patch: AssetRecord, section: AssetSettingsSection): AssetRecord {
  const values = section === "character" ? copyAssetCharacterSettings({}, patch).values : copyAssetStyleSettings({}, patch).values;
  const keys = new Set([...Object.keys(values), section === "character" ? "character" : "asset_style_preset_id"]);
  return Object.fromEntries(Object.entries(patch).filter(([key]) => !keys.has(key)));
}
