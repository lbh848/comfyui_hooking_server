export type RepBatchImage = { filename: string; url: string };
export type RepBatchProfile = {
    clientKey: string; profileId: string; profileLabel: string; rep0: string;
    isNew: boolean; useOwnSearch: boolean; ownSearch: string; manualTarget: string; removePending: boolean;
};
export type RepBatchRow = { charName: string; profiles: RepBatchProfile[]; baseOrder: RepBatchImage[]; loaded: boolean; error: string; maxCards: number };
export type RepBatchPlan = {
    items: { char_name: string; visual_card_id: string; filename?: string; manual_override?: boolean; create_profile?: boolean; profile_label?: string; remove_profile?: boolean }[];
    skipProtected: string[]; skipNoImage: string[]; duplicateGroups: string[];
};
declare global {
    const ComfyRepBatch: {
        profilesFromState(profiles: Record<string, any>[]): RepBatchProfile[];
        characterPalette(charName: string): { background: string; header: string; border: string };
        runLimited(tasks: (() => Promise<void>)[], limit: number): Promise<void>;
        computeDisplay(row: RepBatchRow, profile: RepBatchProfile, search: string, limit?: number): RepBatchImage[];
        capTen(images: RepBatchImage[], row: RepBatchRow, profile: RepBatchProfile, limit?: number): RepBatchImage[];
        target(row: RepBatchRow, search: string): string;
        resolvedTarget(row: RepBatchRow, profile: RepBatchProfile, search: string): string;
        plan(rows: RepBatchRow[], commonSearch: string, mode: "protect" | "push"): RepBatchPlan;
    };
}
