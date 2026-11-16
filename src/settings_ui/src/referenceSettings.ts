import type { AssetRecord } from "./asset";

export const referencePreprocessOptions = [
  ["none", "전처리 없음"], ["lineart", "선만 따기(흑백)"],
  ["lineart_color", "선만 따기(원본 색상)"], ["grayscale", "흑백화"],
] as const;
export const referenceDeliveryOptions = [
  ["none", "대표 이미지만"], ["face_only", "얼굴 이미지만"], ["face", "대표 + 얼굴 이미지"],
  ["representative", "대표 + 두 번째 대표 이미지"], ["stage", "1st·HD·HF에는 대표 / FD·ED에는 얼굴 이미지"],
] as const;
export const referenceStageDeliveryOptions = [
  ...referenceDeliveryOptions.filter(([value]) => value !== "stage"),
  ["llm", "LLM이 결정"],
] as const;

export function referenceStageDelivery(settings: AssetRecord, stage: "primary" | "detailer"): string {
  const legacy = settings.snapref_secondary_reference || "none";
  return settings[`snapref_${stage}_reference`] || (legacy === "stage" ? stage === "detailer" ? "face_only" : "none" : legacy);
}

export function assetReferenceSettings(slot: AssetRecord): AssetRecord {
  // Match the illustration migration from the old logarithmic strength slider.
  const strength = slot.snapref_strength ?? (slot.snapref_strength_scale === "linear" ? 1 : 0);
  return {
    snapref_strength: slot.snapref_strength_scale === "linear" ? strength : Math.min(2, Math.exp(strength)),
    snapref_strength_scale: "linear", snapref_fidelity: slot.snapref_fidelity ?? 1,
    snapref_mode: slot.snapref_mode || "char", snapref_megapixels: slot.snapref_megapixels ?? 1,
    snapref_preprocess: slot.snapref_preprocess || "none", snapref_secondary_reference: slot.snapref_secondary_reference || "none",
  };
}
