export type AssetRecord = Record<string, any>;
export type AssetCapabilities = { type: string; anima: boolean; ilxl: boolean; dual: boolean; pose: boolean; ipadapter: boolean; faceLora: boolean; seed: boolean };
declare global {
  const ComfyAssetPrompt: {
    normalizeAssetWorkflowType(value?: string): string;
    getAssetWorkflowCapabilities(value?: string): AssetCapabilities;
    getAssetPresetValues(tags: AssetRecord, category: string): AssetRecord;
    filterAssetLorasForCapabilities(items: AssetRecord[], capabilities: AssetCapabilities): AssetRecord[];
    migrateBatchChainFields(slot: AssetRecord): void;
    buildAssetPromptData(slot: AssetRecord, tags: AssetRecord, loadPose: (id: string) => Promise<AssetRecord>): Promise<{positive: string; negative: string; positiveLength: number; document?: AssetRecord} | null>;
    assetMaskPlacements(regions: { x: number; y: number; width: number; height: number }[], slots: AssetRecord[], tags: AssetRecord): string[];
    buildAssetMaskPromptData(slots: AssetRecord[], tags: AssetRecord, layout: AssetRecord): Promise<AssetRecord>;
  };
}
