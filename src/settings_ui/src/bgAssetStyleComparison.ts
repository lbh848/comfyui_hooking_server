import {
  applyBgAssetStyleSettings,
  assetCapabilities,
  assetModelPreset,
  assetPresetFieldVisible,
  bgAssetPresetFields,
  copyBgAssetStyleSettings,
  getAssetPresetValues,
  newBgAssetSlot,
  type AssetRecord,
} from "./bgAsset";

export const bgAssetStyleLabels: Record<string, string> = {
  ...Object.fromEntries(
    bgAssetPresetFields
      .filter(field => field[5] === "style")
      .flatMap(([key, label, , raw]) => [[key, `${label} 프리셋`], [raw, `${label} 태그`]]),
  ),
  img_w: "이미지 너비",
  img_h: "이미지 높이",
  seed: "시드",
  hrf_anima: "ANIMA 고해상도 보정",
  hrf_sdxl: "ILXL 고해상도 보정",
  hrf_moss: "MOSS 고해상도 보정",
  hrf_size: "확대 배율",
  hrf_restore_size: "원본 크기 복원",
  style_lora_enabled: "스타일 LoRA 사용",
  style_lora_list: "스타일 LoRA",
};

export function displayBgStyleValue(value: unknown): string {
  if (value == null || value === "" || Array.isArray(value) && !value.length) return "없음";
  if (typeof value === "boolean") return value ? "사용" : "사용 안 함";
  if (Array.isArray(value) && value.every(item => typeof item === "string")) return value.join(", ") || "없음";
  return typeof value === "object" ? JSON.stringify(value, null, 2) : String(value);
}

export function compareBgAssetStyleSettings(before: AssetRecord, after: AssetRecord, tags: AssetRecord = {}) {
  const snapshot = (draft: AssetRecord) => copyBgAssetStyleSettings(newBgAssetSlot(draft.defaults), newBgAssetSlot(draft.slot));
  const previous = snapshot(before);
  const current = snapshot(after);
  const rows: { section: "defaults" | "values"; key: string; before: unknown; after: unknown }[] = [];
  for (const section of ["defaults", "values"] as const) {
    const normalize = (source: AssetRecord, key: string): unknown => {
      const field = bgAssetPresetFields.find(item => item[3] === key);
      if (field) return displayBgStyleValue(source[key] ?? getAssetPresetValues(tags, field[2])[source[field[0]]] ?? []);
      if (key.startsWith("hrf_") && key !== "hrf_size" || key.endsWith("_enabled")) return !!source[key];
      return source[key] ?? "";
    };
    for (const key of Object.keys(previous[section])) {
      const from = normalize(previous[section], key);
      const to = normalize(current[section], key);
      if (JSON.stringify(from) !== JSON.stringify(to)) rows.push({ section, key, before: from, after: to });
    }
  }
  return rows;
}

export function compareBgAssetStyleSwitchEdits(draft: AssetRecord, selected: AssetRecord, tags: AssetRecord = {}) {
  const baseline = applyBgAssetStyleSettings(draft, selected);
  return compareBgAssetStyleSettings(baseline, draft, tags);
}

export function compareBgAssetStyleGeneration(
  defaults: AssetRecord,
  slot: AssetRecord,
  preset: AssetRecord,
  tags: AssetRecord,
  selection: AssetRecord | null,
) {
  const current = newBgAssetSlot(slot);
  const baseline = newBgAssetSlot(preset.settings.values);
  const capabilities = assetCapabilities(assetModelPreset(current, selection)?.config.asset_workflow_type || current.asset_workflow_type);
  const rows: { key: string; before: unknown; after: unknown; changed: boolean }[] = [];
  function add(key: string, before: unknown, after: unknown) {
    rows.push({ key, before, after, changed: displayBgStyleValue(before) !== displayBgStyleValue(after) });
  }
  for (const [key, , category, raw, model, section] of bgAssetPresetFields) {
    if (section !== "style" || !assetPresetFieldVisible(model, capabilities)) continue;
    const values = getAssetPresetValues(tags, category);
    add(raw, defaults[raw] ?? values[defaults[key]] ?? [], current[raw] ?? values[current[key]] ?? []);
  }
  const options = copyBgAssetStyleSettings({}, baseline).values;
  for (const key of Object.keys(options)) {
    if (bgAssetPresetFields.some(field => field[0] === key || field[3] === key)) continue;
    if ((key.startsWith("anima_") || key === "hrf_anima") && !capabilities.anima) continue;
    if ((key.startsWith("sdxl_") || key === "hrf_sdxl") && !capabilities.ilxl) continue;
    if (key === "hrf_moss" && !capabilities.moss) continue;
    if (key === "seed" && !capabilities.seed) continue;
    if ((key === "hrf_size" || key === "hrf_restore_size") && ![baseline, current].some(value =>
      capabilities.anima && value.hrf_anima || capabilities.ilxl && value.hrf_sdxl || capabilities.moss && value.hrf_moss)) continue;
    if (key === "style_lora_list" && !baseline.style_lora_enabled && !current.style_lora_enabled) continue;
    const value = (source: AssetRecord) => {
      if (key.startsWith("hrf_") && key !== "hrf_size" || key.endsWith("_enabled")) return !!source[key];
      return source[key];
    };
    add(key, value(baseline), value(current));
  }
  return rows;
}

export function restoreBgAssetStyleGeneration(rows: ReturnType<typeof compareBgAssetStyleGeneration>): AssetRecord {
  return structuredClone(Object.fromEntries(rows.filter(row => row.changed).map(row => [row.key, row.before])));
}
