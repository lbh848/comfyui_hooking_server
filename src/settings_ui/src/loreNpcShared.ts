import type { MutableRefObject } from "react";
import type { BotApi } from "./BotApiScope";

export type LoreNpcCard = {
  gender_tag: string;
  appearance: string[];
  default_outfit: string[];
  face_tags: string[];
  eye_tags: string[];
  visual_context: string;
  name_tag?: string;
};

export type LoreNpc = {
  name: string;
  aliases: string[];
  description: string;
  entry_indexes: number[];
  card: LoreNpcCard;
};

export type LoreImageGroup = {
  id: number;
  group: string;
  count: number;
  samples: string[];
  indexes: number[];
  character: string | null;
  reason: string;
};

export type AnalysisWarning = {
  batch: number;
  source: string;
  title: string;
  issues: string[];
};

export type LoreOperation = {
  controller: AbortController;
  requestId: string;
};

export const LORE_BATCH_CHAR_OPTIONS = [5000, 10000, 20000, 40000, 80000] as const;

export function buildLoreVisualCardPayload(candidate: LoreNpc) {
  const nameTag = (candidate.card.name_tag || "").trim();
  const appearance = nameTag && !candidate.card.appearance.some(tag => tag.toLowerCase() === nameTag.toLowerCase())
    ? [nameTag, ...candidate.card.appearance]
    : candidate.card.appearance;
  return {
    character: candidate.name,
    data: {
      name: candidate.name,
      default_visual_profile_id: "card_1",
      profiles: [{
        id: "card_1",
        label: "카드 1",
        selection_guide: candidate.description || "",
        visual_context: candidate.card.visual_context || "",
        aliases: candidate.aliases || [],
        appearance,
        default_outfit: candidate.card.default_outfit || [],
        render_overrides: {
          gender_tag: candidate.card.gender_tag || "1girl",
          face_tags: (candidate.card.face_tags || []).join(", "),
          eye_tags: (candidate.card.eye_tags || []).join(", "),
          use_profile_embedding: false,
          use_image_name_tag: Boolean(nameTag),
          ...(nameTag ? { image_name_tag: nameTag } : {}),
          rep_images: [],
        },
      }],
    },
  };
}

export function mergeLoreNpcCandidates(
  previousCandidates: readonly LoreNpc[],
  foundCandidates: readonly LoreNpc[],
  keepPrevious: boolean,
  existingNames?: ReadonlySet<string>,
): LoreNpc[] {
  const canonicalExisting = new Map<string, string>();
  if (existingNames) {
    for (const existing of existingNames) canonicalExisting.set(existing.toLowerCase(), existing);
  }
  const merged = new Map<string, LoreNpc>();
  if (keepPrevious) {
    for (const candidate of previousCandidates) {
      merged.set(candidate.name.toLowerCase(), candidate);
    }
  }
  for (const raw of foundCandidates) {
    const key = raw.name.toLowerCase();
    const canonicalName = canonicalExisting.get(key) || raw.name;
    const candidate = canonicalName === raw.name ? raw : { ...raw, name: canonicalName };
    const old = merged.get(key);
    merged.set(key, old ? {
      ...candidate,
      name: canonicalExisting.get(key) || old.name,
      aliases: [...new Set([...old.aliases, ...candidate.aliases])],
      entry_indexes: [...new Set([...old.entry_indexes, ...candidate.entry_indexes])],
    } : candidate);
  }
  return [...merged.values()];
}

export async function requestLoreNpc(
  api: BotApi,
  botName: string,
  operation: LoreOperation,
  action: "generate" | "register" | "classify_images",
  body: Record<string, unknown>,
) {
  operation.controller.signal.throwIfAborted();
  operation.requestId = crypto.randomUUID();
  try {
    // Keep the response open so Stop can return already analyzed candidates or groups.
    return await api.request(`bot_mode/lore_npcs/${action}`, { ...body, bot_name: botName, operation_id: operation.requestId });
  } finally {
    operation.requestId = "";
  }
}

export async function cancelLoreNpcOperation(
  api: BotApi,
  botName: string,
  operation: LoreOperation,
  activeRef: MutableRefObject<LoreOperation | null>,
) {
  operation.controller.abort();
  const id = operation.requestId;
  // A request can still be waiting for the bot write lock when Stop is pressed.
  while (id && activeRef.current === operation && operation.requestId === id) {
    const data = await api.request("illustration/lore_npcs/cancel", { bot_name: botName, operation_id: id });
    if (data.found) break;
    await new Promise(resolve => window.setTimeout(resolve, 500));
  }
}
