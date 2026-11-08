export type BotBatchRule = { action: "split_by" | "strip_suffix" | "strip_prefix" | "replace" | "strip"; separator?: string; take?: string | number; pattern?: string; from?: string; to?: string };
export type BotBatchEntry = { file: File; name: string; action: "added" | "replaced" | "unchanged"; existingFilename?: string; preflightIndex: number };
export type BotBatchResult = { added: number; uploaded: number; replaced: number; skipped: number; errors: string[] };
export type BotBatchFile = { file: File; path: string };
declare global {
    const ComfyBotBatch: {
        getStorageNameError(value: string, label?: string): string;
        applyRules(filename: string, rules: BotBatchRule[]): string;
        fileSha256(file: File): Promise<string>;
        runConcurrent<T>(items: T[], process: (item: T, index: number) => Promise<void>, options?: { signal?: AbortSignal; concurrency?: number }): Promise<void>;
        collectImageFiles(source: FileList | DataTransfer, signal: AbortSignal, onProgress?: (count: number) => void): Promise<{ files: BotBatchFile[]; skipped: number }>;
        importCharacters(options: {
            botName: string; names: string[]; entries: BotBatchEntry[]; preflightId: string;
            request: (path: string, body: Record<string, unknown> | FormData) => Promise<Record<string, any>>;
            onProgress?: (text: string) => void;
        }): Promise<BotBatchResult>;
    };
}
