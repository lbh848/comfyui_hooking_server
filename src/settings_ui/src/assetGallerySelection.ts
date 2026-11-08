import { copyAssetCharacterSettings, getAssetPresetValues, loadAssetCharacterSettings, type AssetRecord } from "./asset";

export type AssetGalleryCombination = { outfit: string; expression: string };

function presetFromFolder(values: AssetRecord, folder: string) {
  if (Object.hasOwn(values, folder)) return folder;
  // Match the original gallery and the server's _safe_dirname normalization.
  const matches = Object.keys(values).filter(name => name.replace(/[^\p{L}\p{N} _\-.]/gu, "").trim() === folder);
  return matches.length === 1 ? matches[0] : "";
}

export function applyAssetGallerySelection(draft: AssetRecord, character: string, combination: AssetGalleryCombination | null, tags: AssetRecord): AssetRecord {
  if (!character) return draft;
  const saved = draft.slot.character === character ? null : loadAssetCharacterSettings(character, tags);
  const defaults = saved ? { ...draft.defaults, ...saved.defaults } : draft.defaults;
  const slot = { ...draft.slot, ...saved?.values, character };
  if (combination) {
    for (const [key, category, raw] of [["outfit", "outfits", "raw_outfit_tags"], ["expression", "expressions", "raw_expression_tags"]] as const) {
      const values = getAssetPresetValues(tags, category);
      slot[key] = presetFromFolder(values, combination[key]);
      // Replace applied tags too; changing only the label would keep the previous prompt.
      slot[raw] = structuredClone(values[slot[key]] || []);
    }
  }
  const baseline = saved || copyAssetCharacterSettings(draft.defaults, draft.slot);
  const changed = JSON.stringify(copyAssetCharacterSettings(defaults, slot)) !== JSON.stringify(baseline);
  return { ...draft, defaults, slot, characterSettingsDirty: changed || !saved && !!draft.characterSettingsDirty };
}
