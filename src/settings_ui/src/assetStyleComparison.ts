import { applyAssetStyleSettings, assetCapabilities, assetModelPreset, assetPresetFields, copyAssetStyleSettings, getAssetPresetValues, newAssetSlot, type AssetRecord } from "./asset";

export const assetStyleLabels: Record<string, string> = {
  ...Object.fromEntries(assetPresetFields.filter(field => field[5] === "style").flatMap(([key, label, , raw]) => [[key, `${label} 프리셋`], [raw, `${label} 태그`]])),
  img_w: "이미지 너비", img_h: "이미지 높이", seed: "시드",
  mask_area_limit_enabled: "마스크 영역 제한", mask_box: "마스크 위치와 크기",
  hrf_anima: "ANIMA 고해상도 보정", hrf_sdxl: "ILXL 고해상도 보정", hrf_size: "확대 배율", hrf_restore_size: "원본 크기 복원",
  anima_fd_enabled: "ANIMA 얼굴 보정", anima_hd_enabled: "ANIMA 손 보정", anima_ed_enabled: "ANIMA 눈 보정",
  sdxl_fd_enabled: "ILXL 얼굴 보정", sdxl_hd_enabled: "ILXL 손 보정", sdxl_ed_enabled: "ILXL 눈 보정",
  style_lora_enabled: "스타일 LoRA 사용", style_lora_list: "스타일 LoRA",
};

// Include defaults and inactive options: switching a saved style replaces both.
export function compareAssetStyleSettings(before: AssetRecord, after: AssetRecord, tags: AssetRecord = {}) {
  const snapshot = (draft: AssetRecord) => copyAssetStyleSettings(newAssetSlot(draft.defaults), newAssetSlot(draft.slot));
  const previous = snapshot(before), current = snapshot(after);
  const rows: { section: "defaults" | "values"; key: string; before: unknown; after: unknown }[] = [];
  for (const section of ["defaults", "values"] as const) {
    const normalize = (source: AssetRecord, key: string): unknown => {
      const field = assetPresetFields.find(field => field[3] === key);
      if (field) return displayStyleValue(source[key] ?? getAssetPresetValues(tags, field[2])[source[field[0]]] ?? []);
      if (key.startsWith("hrf_") && key !== "hrf_size" || key.endsWith("_enabled")) return !!source[key];
      return source[key] ?? "";
    };
    for (const key of Object.keys(previous[section])) {
      const from = normalize(previous[section], key), to = normalize(current[section], key);
      if (JSON.stringify(from) !== JSON.stringify(to)) rows.push({ section, key, before: from, after: to });
    }
  }
  return rows;
}

export function compareAssetStyleSwitchEdits(draft: AssetRecord, selected: AssetRecord, tags: AssetRecord = {}) {
  const baseline = applyAssetStyleSettings(draft, selected);
  return compareAssetStyleSettings(baseline, draft, tags);
}

export function displayStyleValue(value: unknown, key?: string): string {
  if (value == null || value === "" || Array.isArray(value) && !value.length) return "없음";
  if (key === "mask_box" && typeof value === "object") {
    const box = value as AssetRecord;
    const percent = (n: number) => `${Math.round(n * 10000) / 100}%`;
    return `왼쪽 ${percent(box.x)} · 위 ${percent(box.y)} · 너비 ${percent(box.width)} · 높이 ${percent(box.height)}`;
  }
  if (typeof value === "boolean") return value ? "사용" : "사용 안 함";
  if (Array.isArray(value) && value.every(item => typeof item === "string")) return value.join(", ") || "없음";
  return typeof value === "object" ? JSON.stringify(value, null, 2) : String(value);
}

// Tag preset selection establishes the baseline. Only applied values are compared;
// saving or renaming a preset is separate from changing this generation's values.
export function compareAssetStyleGeneration(defaults: AssetRecord, slot: AssetRecord, preset: AssetRecord,
  tags: AssetRecord, selection: AssetRecord | null) {
  const current = newAssetSlot(slot);
  const baseline = newAssetSlot(preset.settings.values);
  const capabilities = assetCapabilities(assetModelPreset(current, selection)?.config.asset_workflow_type || current.asset_workflow_type);
  const rows: { key: string; before: unknown; after: unknown; changed: boolean }[] = [];
  function add(key: string, before: unknown, after: unknown) {
    rows.push({ key, before, after, changed: displayStyleValue(before) !== displayStyleValue(after) });
  }
  for (const [key, , category, raw, model, section] of assetPresetFields) {
    if (section !== "style" || model && !capabilities[model]) continue;
    const values = getAssetPresetValues(tags, category);
    add(raw, defaults[raw] ?? values[defaults[key]] ?? [], current[raw] ?? values[current[key]] ?? []);
  }
  // These controls have no separate tag-preset column; the selected style supplies
  // their baseline, including defaults for older presets with omitted options.
  const options = copyAssetStyleSettings({}, baseline).values;
  for (const key of Object.keys(options)) {
    if (assetPresetFields.some(field => field[0] === key || field[3] === key)) continue;
    if ((key.startsWith("anima_") || key === "hrf_anima") && !capabilities.anima) continue;
    if ((key.startsWith("sdxl_") || key === "hrf_sdxl") && !capabilities.ilxl) continue;
    if (key === "seed" && !capabilities.seed) continue;
    if ((key === "hrf_size" || key === "hrf_restore_size") && ![baseline, current].some(value =>
      capabilities.anima && value.hrf_anima || capabilities.ilxl && value.hrf_sdxl)) continue;
    if (key === "style_lora_list" && !baseline.style_lora_enabled && !current.style_lora_enabled) continue;
    const value = (source: AssetRecord) => {
      if (key.startsWith("hrf_") && key !== "hrf_size" || key.endsWith("_enabled")) return !!source[key];
      return source[key];
    };
    add(key, value(baseline), value(current));
  }
  return rows;
}

export function restoreAssetStyleGeneration(rows: ReturnType<typeof compareAssetStyleGeneration>): AssetRecord {
  return structuredClone(Object.fromEntries(rows.filter(row => row.changed).map(row => [row.key, row.before])));
}
