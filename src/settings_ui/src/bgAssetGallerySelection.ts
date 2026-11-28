import { loadBgThemeData, pickBgPlaceAndTime, type AssetRecord } from "./bgAsset";

export type BgAssetGalleryCombination = { place: string; atmosphere?: string };

function matchFolderName(keys: string[], folder: string): string {
  if (keys.includes(folder)) return folder;
  const matches = keys.filter(name => name.replace(/[^\p{L}\p{N} _\-.]/gu, "").trim() === folder);
  return matches.length === 1 ? matches[0] : folder;
}

export function applyBgAssetGallerySelection(
  draft: AssetRecord,
  theme: string,
  combination: BgAssetGalleryCombination | null,
  tags: AssetRecord,
): AssetRecord {
  if (!theme) return draft;
  const themeData = loadBgThemeData(theme, tags);
  const placeNames = Object.keys(themeData.places);
  let place = draft.slot.theme === theme ? (draft.slot.place || "") : "";
  let time = draft.slot.theme === theme ? (draft.slot.time || "") : "";

  if (combination?.place) {
    const matchedPlace = matchFolderName(placeNames, combination.place);
    const timeNames = Object.keys(themeData.places[matchedPlace]?.times || {});
    place = matchedPlace;
    if (combination.atmosphere) {
      time = matchFolderName(timeNames, combination.atmosphere);
    } else if (Object.hasOwn(themeData.places, matchedPlace)) {
      time = pickBgPlaceAndTime(themeData, matchedPlace, time).time;
    } else {
      time = "";
    }
  } else {
    const picked = pickBgPlaceAndTime(themeData, place, time);
    place = picked.place;
    time = picked.time;
  }

  return {
    ...draft,
    themeData,
    themeSettingsDirty: false,
    slot: {
      ...draft.slot,
      theme,
      place,
      time,
      atmosphere: time,
    },
  };
}
