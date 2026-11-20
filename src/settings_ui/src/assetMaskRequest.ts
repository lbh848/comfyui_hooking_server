import { buildRegionalAssetRequest, copyAssetCharacterSettings, type AssetRecord } from "./asset";
import { assetReferenceSettings } from "./referenceSettings";

export type MaskBox = { x: number; y: number; width: number; height: number };
export type MaskRegion = { id: string; slot: AssetRecord; box: MaskBox | null };

export function boxBetween(start: { x: number; y: number }, end: { x: number; y: number }): MaskBox {
  const clamp = (value: number) => Math.max(0, Math.min(1, value));
  const x = Math.min(clamp(start.x), clamp(end.x)), y = Math.min(clamp(start.y), clamp(end.y));
  return { x, y, width: Math.max(clamp(start.x), clamp(end.x)) - x, height: Math.max(clamp(start.y), clamp(end.y)) - y };
}

export function assetMaskPlacements(regions: MaskRegion[], tags: AssetRecord): string[] {
  const boxes = regions.map(region => region.box);
  if (!boxes.length || boxes.some(box => !box || box.width <= 0 || box.height <= 0)) return [];
  return ComfyAssetPrompt.assetMaskPlacements(boxes as MaskBox[], regions.map(region => region.slot), tags);
}

export async function buildAssetMaskRequest(base: AssetRecord, regions: MaskRegion[], composition: string,
  tags: AssetRecord, settings: AssetRecord, styles: AssetRecord) {
  if (!regions.length || regions.length > 3) throw new Error("캐릭터 영역은 1~3개까지 사용할 수 있습니다.");
  if (regions.some(region => !region.box || region.box.width <= 0 || region.box.height <= 0)) throw new Error("각 캐릭터의 사각형 영역을 그려주세요.");
  const slots: AssetRecord[] = regions.map(region => ({ ...base, ...copyAssetCharacterSettings({}, region.slot).values,
    ...assetReferenceSettings(region.slot),
    character: region.slot.character, mask_area_limit_enabled: false }));
  if (slots.some(slot => !slot.character)) throw new Error("각 영역의 캐릭터를 선택하세요.");
  const boxes = regions.map(region => ({ name: region.id, character: region.slot.character, ...region.box }));
  return buildRegionalAssetRequest(slots, boxes, composition, tags, settings, styles);
}
