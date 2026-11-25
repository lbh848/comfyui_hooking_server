import type { ImportedGlobalNote } from "./charxImages";

export type CharacterLoreEntry = {
  entry_id: string; source: string; title: string; content: string;
  keys: string[]; secondary_keys: string[];
  enabled: boolean; constant: boolean; selective: boolean; use_regex: boolean;
};

const ASSET_INSTRUCTION_LORE_RE = /<img\s+src|\{\{(?:img|raw)::|\bimage\s+tag|\basset\s+prompt|inserting\s+character\s+visual|image\s+output\s+(?:setting|instruction|guidance|format|rule)|image\s+command\s+instruction|에셋\s*출력|이미지\s*태그|이미지\s*출력/i;

export function extractLoreAssetInstructions(entries: readonly CharacterLoreEntry[]): ImportedGlobalNote[] {
  const bySource = new Map<string, CharacterLoreEntry[]>();
  for (const entry of entries) {
    if (!entry.enabled || !entry.constant) continue;
    const content = entry.content.trim();
    if (!content) continue;
    if (entry.title.includes("__ASSET_MAID_DATA__") || /^\s*@@dont_activate\b/i.test(content)) continue;
    if (!ASSET_INSTRUCTION_LORE_RE.test(`${entry.title}\n${content}`)) continue;
    const list = bySource.get(entry.source) || [];
    list.push(entry);
    bySource.set(entry.source, list);
  }
  const notes: ImportedGlobalNote[] = [];
  for (const [source, matched] of bySource.entries()) {
    const titles = matched.map(entry => entry.title.trim()).filter(Boolean).slice(0, 4);
    notes.push({
      source: `${source} · 로어북 에셋 지침${titles.length ? ` (${titles.join(", ")})` : ""}`,
      content: matched.map(entry => entry.content.trim()).join("\n\n"),
    });
  }
  return notes;
}

function loreKeys(value: unknown): string[] {
  const items = Array.isArray(value) ? value : typeof value === "string" ? value.split(",") : [];
  return [...new Set(items.filter((item): item is string => typeof item === "string").map(item => item.trim()).filter(Boolean))];
}

export function readCharacterLore(value: unknown, source: string): CharacterLoreEntry[] {
  if (!Array.isArray(value)) return [];
  const entries = value.flatMap((entry, index) => {
    if (!entry || typeof entry !== "object" || entry.mode === "folder") return [];
    const title = String(entry.name || entry.comment || "");
    const keys = loreKeys(entry.keys ?? entry.key);
    const secondary_keys = loreKeys(entry.secondary_keys ?? entry.secondkey);
    const content = typeof entry.content === "string" ? entry.content : "";
    if (!title && !keys.length && !content) return [];
    const entryId = entry.id == null ? "" : String(entry.id);
    return [{ entry_id: entryId.trim() ? entryId : JSON.stringify([title || index, keys, secondary_keys]), source,
      title: title || keys[0] || `로어북 ${index + 1}`, content, keys, secondary_keys,
      enabled: entry.enabled !== false, constant: entry.constant === true || entry.alwaysActive === true || entry.mode === "constant",
      selective: entry.selective === true, use_regex: entry.use_regex === true || entry.useRegex === true }];
  });
  const reservedIds = new Set(entries.map(entry => entry.entry_id));
  const seenIds = new Set<string>();
  for (const entry of entries) {
    const entryId = entry.entry_id;
    if (seenIds.has(entryId)) {
      // Same-title entries can have different contents; preserve every occurrence.
      let occurrence = 1;
      do { entry.entry_id = JSON.stringify([entryId, occurrence++]); } while (reservedIds.has(entry.entry_id));
      reservedIds.add(entry.entry_id);
    }
    seenIds.add(entry.entry_id);
  }
  return entries;
}

export const characterLoreId = (entry: Pick<CharacterLoreEntry, "source" | "entry_id">): string => JSON.stringify([entry.source, entry.entry_id]);
const normalizedName = (value: string): string => value.normalize("NFKC").toLowerCase().replace(/[^\p{L}\p{N}]+/gu, " ").trim();

export function matchCharacterLore(entry: CharacterLoreEntry, characterNames: readonly string[]): string[] {
  const keys = entry.use_regex ? [] : [...entry.keys, ...entry.secondary_keys].map(normalizedName);
  const title = ` ${normalizedName(entry.title)} `;
  const matches = characterNames.filter(name => {
    const normalized = normalizedName(name);
    return normalized && (keys.includes(normalized) || title.includes(` ${normalized} `));
  });
  // Shared entries and partial-name collisions need an explicit choice in the preview.
  return matches.length === 1 ? matches : [];
}
