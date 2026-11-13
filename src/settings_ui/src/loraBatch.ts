import type { IllustrationData } from "./illustration";
import { selectedLoraPath } from "./loraPicker";

export type LoraBatchMode = "add" | "replace" | "clear";

export function batchLoras(current: IllustrationData[], next: IllustrationData[], mode: LoraBatchMode): IllustrationData[] {
  if (mode === "clear") return [];
  const after = mode === "replace" && next.length ? [] : [...current];
  for (const item of next) {
    const path = selectedLoraPath(item);
    const index = after.findIndex(existing => selectedLoraPath(existing) === path);
    if (index < 0) after.push(structuredClone(item));
    else {
      const strength = item.strength ?? item.str ?? 0.5;
      // Keep the card's other settings when updating an existing LoRA's strength.
      if (Number(after[index].strength ?? after[index].str ?? 0.5) !== strength) {
        after[index] = { ...after[index], strength };
      }
    }
  }
  return after;
}
