export type CharacterLoreEntry = {
  entry_id: string; source: string; title: string; content: string;
  keys: string[]; secondary_keys: string[];
  enabled: boolean; constant: boolean; selective: boolean; use_regex: boolean;
};

function loreKeys(value: unknown): string[] {
  const items = Array.isArray(value) ? value : typeof value === "string" ? value.split(",") : [];
  return [...new Set(items.filter((item): item is string => typeof item === "string").map(item => item.trim()).filter(Boolean))];
}

export function readCharacterLore(value: unknown, source: string): CharacterLoreEntry[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((entry, index) => {
    if (!entry || typeof entry !== "object" || entry.mode === "folder") return [];
    const title = String(entry.name || entry.comment || "");
    const keys = loreKeys(entry.keys ?? entry.key);
    const secondary_keys = loreKeys(entry.secondary_keys ?? entry.secondkey);
    const content = typeof entry.content === "string" ? entry.content : "";
    if (!title && !keys.length && !content) return [];
    return [{ entry_id: String(entry.id ?? JSON.stringify([title || index, keys, secondary_keys])), source,
      title: title || keys[0] || `로어북 ${index + 1}`, content, keys, secondary_keys,
      enabled: entry.enabled !== false, constant: entry.constant === true || entry.alwaysActive === true || entry.mode === "constant",
      selective: entry.selective === true, use_regex: entry.use_regex === true || entry.useRegex === true }];
  });
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
