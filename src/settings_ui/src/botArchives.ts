import type { BotApi } from "./BotApiScope";
import type { CharacterLoreEntry } from "./characterLore";
import { query } from "./illustration";

export type BotArchive = {
  id: string; kind: "bot" | "module"; filename: string; size: number; saved_at: string;
  lore_entries: CharacterLoreEntry[];
};

export async function saveBotArchives(api: BotApi, botName: string, botFiles: readonly File[], moduleFiles: readonly File[],
  loreEntries: CharacterLoreEntry[], onProgress?: (message: string, completed: number) => void, signal?: AbortSignal) {
  const archives: BotArchive[] = [];
  for (const [kind, files, label] of [["bot", botFiles, "봇"], ["module", moduleFiles, "모듈"]] as const) {
    for (const file of files) {
      signal?.throwIfAborted();
      onProgress?.(`${file.name} 원본 저장 중…`, archives.length);
      const body = new FormData();
      body.append("lore_entries", JSON.stringify(loreEntries.filter(entry => entry.source === `${label}/${file.name}`)));
      body.append("file", file);
      const result = await api.request(`bot_mode/archive?${query({ bot_name: botName, kind })}`, body, signal);
      archives.push(result.archive as BotArchive);
      onProgress?.(`${file.name} 원본 저장 완료`, archives.length);
    }
  }
  return archives;
}
