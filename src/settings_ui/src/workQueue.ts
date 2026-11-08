import { loraRequest, type LoraData } from "./internalLora";

export const QUEUE_HISTORY_LIMIT = 10;
export const isFinishedQueueItem = (item: LoraData) => ["completed", "failed", "cancelled"].includes(item.status);

export function queueHistoryItems(items: LoraData[]): LoraData[] {
  return items.filter(isFinishedQueueItem)
    .sort((a, b) => (b.completed_at || b.created_at || 0) - (a.completed_at || a.created_at || 0))
    .slice(0, QUEUE_HISTORY_LIMIT);
}

export function watchWorkQueue(onChange: (queue: LoraData) => void, onError: (cause: unknown) => void): () => void {
  const controller = new AbortController();
  let timer: number;
  let reconnect: number;
  let socket: WebSocket | null = null;
  let revision = 0;
  let latest: LoraData | null = null;
  const receive = (queue: LoraData) => {
    if (controller.signal.aborted) return;
    revision++;
    latest = queue;
    onChange(queue);
  };
  const connect = () => {
    socket = new WebSocket(`${location.protocol === "https:" ? "wss:" : "ws:"}//${location.host}/api/illust/events?schema=2`);
    socket.onmessage = event => {
      if (controller.signal.aborted) return;
      const value = JSON.parse(event.data);
      if (value.type === "queue_updated") receive(value.data);
      if (value.type === "queue_progress" && latest?.items?.some((item: LoraData) => item.id === value.data.item_id && item.status === "processing")) {
        receive({ ...latest, items: latest.items.map((item: LoraData) => item.id === value.data.item_id && item.status === "processing"
          ? { ...item, progress: value.data.progress, progress_detail: value.data.detail, subtasks: value.data.subtasks ?? item.subtasks } : item) });
      }
    };
    socket.onclose = () => {
      if (!controller.signal.aborted) reconnect = window.setTimeout(connect, 3000);
    };
  };
  const poll = async () => {
    const before = revision;
    try {
      const queue = await loraRequest("queue/status?monitor=1", undefined, controller.signal);
      if (before === revision) receive(queue);
    } catch (cause) {
      if (!controller.signal.aborted && before === revision) onError(cause);
    } finally {
      if (!controller.signal.aborted) timer = window.setTimeout(poll, 2000);
    }
  };
  connect();
  void poll();
  return () => { controller.abort(); clearTimeout(timer); clearTimeout(reconnect); socket?.close(); };
}
