export type VisualGuideSuggestion = {
    aliases: string[]; gender_tag: string; selection_guide: string; selection_guide_english: string; evidence: string;
    visual_context: string; visual_context_english: string; visual_context_status: string;
    visual_context_error: string; visual_context_image: string; confidence: string;
};
export type VisualGuideTarget = {
    identity: string; character: string; profileId: string; profileLabel: string; profileIndex: number;
    repFilename: string; imageUrl: string; currentAliases: string[]; currentGenderTag: string; currentGuide: string; currentVisualContext: string;
    selected: boolean; focused: boolean; suggestion: VisualGuideSuggestion | null;
};
declare global {
    const ComfyVisualGuide: {
        genderTags: readonly string[];
        identity(character: string, profileId: string): string;
        missing(target: VisualGuideTarget, analyzeImages?: boolean): boolean;
        target(botName: string, character: string, profile: Record<string, any>, index: number, focusIdentity?: string): VisualGuideTarget;
        matches(target: VisualGuideTarget, query: string): boolean;
        merge(targets: VisualGuideTarget[], suggestions: Record<string, any>[]): number;
        metadataItems(targets: VisualGuideTarget[]): Record<string, any>[];
    };
}
