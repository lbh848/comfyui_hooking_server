import type { LoraData } from "./internalLora";
import { subscribeQueueStatus } from "./workshopEvents";

export const QUEUE_HISTORY_LIMIT = 10;
export const isFinishedQueueItem = (item: LoraData) => ["completed", "failed", "cancelled"].includes(item.status);

export function watchCompletedQueueItems(matches: (item: LoraData) => boolean, onCompleted: () => void, onError: (cause: unknown) => void): () => void {
  const completed = new Set<string>();
  return watchWorkQueue(queue => {
    let changed = false;
    for (const item of queue.items || []) {
      if (item.status !== "completed" || !matches(item)) continue;
      const key = JSON.stringify([item.id, item.completed_at]);
      if (completed.has(key)) continue;
      completed.add(key);
      changed = true;
    }
    // Catch a completion racing the initial project load as well as later batches.
    if (changed) onCompleted();
  }, onError);
}

export function queueHistoryItems(items: LoraData[]): LoraData[] {
  return items.filter(isFinishedQueueItem)
    .sort((a, b) => (b.completed_at || b.created_at || 0) - (a.completed_at || a.created_at || 0))
    .slice(0, QUEUE_HISTORY_LIMIT);
}

export function watchWorkQueue(onChange: (queue: LoraData) => void, onError: (cause: unknown) => void): () => void {
  return subscribeQueueStatus(onChange, { onError, activateWorker: true });
}
