import { loraRequest, type LoraData } from "./internalLora";
import { activeLlmStreams, llmLogsRequest, type LlmStream } from "./llmLogs";

export type WorkshopEventMessage = {
  type: string;
  data?: any;
};

type EventSubscriber = {
  onMessage: (message: WorkshopEventMessage) => void;
  onReconnect?: () => void;
  onOpen?: () => void;
};

const eventSubscribers = new Set<EventSubscriber>();
const connectionListeners = new Set<(connected: boolean) => void>();
let eventSocket: WebSocket | null = null;
let eventReconnectTimer: number | null = null;
let eventCloseTimer: number | null = null;
let eventReconnectDelayMs = 1500;
let eventHasConnectedOnce = false;
let eventConnected = false;

function setEventConnected(next: boolean) {
  if (eventConnected === next) return;
  eventConnected = next;
  for (const listener of connectionListeners) {
    try { listener(next); } catch { /* ignore subscriber error */ }
  }
}

function clearEventTimers() {
  if (eventReconnectTimer !== null) {
    window.clearTimeout(eventReconnectTimer);
    eventReconnectTimer = null;
  }
  if (eventCloseTimer !== null) {
    window.clearTimeout(eventCloseTimer);
    eventCloseTimer = null;
  }
}

function connectEventSocket() {
  if (!eventSubscribers.size) return;
  if (eventSocket && (eventSocket.readyState === WebSocket.OPEN || eventSocket.readyState === WebSocket.CONNECTING)) {
    return;
  }
  clearEventTimers();
  const ws = new WebSocket(`${location.protocol === "https:" ? "wss:" : "ws:"}//${location.host}/api/illust/events?schema=2`);
  eventSocket = ws;
  ws.onopen = () => {
    if (eventSocket !== ws) {
      ws.close();
      return;
    }
    const isReconnect = eventHasConnectedOnce;
    eventHasConnectedOnce = true;
    eventReconnectDelayMs = 1500;
    setEventConnected(true);
    for (const subscriber of Array.from(eventSubscribers)) {
      try { subscriber.onOpen?.(); } catch { /* ignore subscriber error */ }
      if (isReconnect) {
        try { subscriber.onReconnect?.(); } catch { /* ignore subscriber error */ }
      }
    }
  };
  ws.onmessage = event => {
    if (eventSocket !== ws) return;
    let parsed: WorkshopEventMessage;
    try {
      parsed = JSON.parse(event.data);
    } catch {
      return;
    }
    for (const subscriber of Array.from(eventSubscribers)) {
      try { subscriber.onMessage(parsed); } catch { /* ignore subscriber error */ }
    }
  };
  ws.onclose = () => {
    if (eventSocket !== ws) return;
    eventSocket = null;
    setEventConnected(false);
    if (!eventSubscribers.size) {
      eventHasConnectedOnce = false;
      return;
    }
    if (document.hidden) return;
    const delay = eventReconnectDelayMs;
    eventReconnectDelayMs = Math.min(5000, Math.round(eventReconnectDelayMs * 1.6));
    eventReconnectTimer = window.setTimeout(() => {
      eventReconnectTimer = null;
      connectEventSocket();
    }, delay);
  };
}

function closeEventSocketIfIdle() {
  if (eventSubscribers.size) return;
  clearEventTimers();
  const ws = eventSocket;
  eventSocket = null;
  eventHasConnectedOnce = false;
  setEventConnected(false);
  ws?.close();
}

export function isWorkshopEventsConnected(): boolean {
  return eventConnected;
}

export function subscribeWorkshopConnection(listener: (connected: boolean) => void): () => void {
  connectionListeners.add(listener);
  return () => { connectionListeners.delete(listener); };
}

export function subscribeWorkshopEvents(
  onMessage: (message: WorkshopEventMessage) => void,
  options?: { onReconnect?: () => void; onOpen?: () => void },
): () => void {
  const subscriber: EventSubscriber = {
    onMessage,
    onReconnect: options?.onReconnect,
    onOpen: options?.onOpen,
  };
  eventSubscribers.add(subscriber);
  if (eventCloseTimer !== null) {
    window.clearTimeout(eventCloseTimer);
    eventCloseTimer = null;
  }
  connectEventSocket();
  return () => {
    eventSubscribers.delete(subscriber);
    if (!eventSubscribers.size && eventCloseTimer === null) {
      eventCloseTimer = window.setTimeout(() => {
        eventCloseTimer = null;
        closeEventSocketIfIdle();
      }, 500);
    }
  };
}

// ─── Shared Work Queue Status Store ────────────────────────────────────

type QueueSubscriber = {
  onChange: (queue: LoraData) => void;
  onError?: (cause: unknown) => void;
  activateWorker: boolean;
};

const queueSubscribers = new Set<QueueSubscriber>();
let latestQueue: LoraData | null = null;
let latestQueueUpdatedAt = 0;
let queueRevision = 0;
let queuePollTimer: number | null = null;
let queuePollController: AbortController | null = null;
let queuePollInFlight = false;
let queueWsUnsubscribe: (() => void) | null = null;

function hasActiveQueueWorkerSubscriber(): boolean {
  for (const subscriber of queueSubscribers) {
    if (subscriber.activateWorker) return true;
  }
  return false;
}

function clearQueuePollTimer() {
  if (queuePollTimer !== null) {
    window.clearTimeout(queuePollTimer);
    queuePollTimer = null;
  }
}

function scheduleQueuePoll() {
  clearQueuePollTimer();
  if (!queueSubscribers.size || document.hidden) return;
  const interval = queueWsUnsubscribe && eventConnected ? 15000 : 3000;
  queuePollTimer = window.setTimeout(() => {
    queuePollTimer = null;
    void pollQueueOnce();
  }, interval);
}

function deliverQueue(queue: LoraData) {
  queueRevision++;
  latestQueue = queue;
  latestQueueUpdatedAt = Date.now();
  for (const subscriber of Array.from(queueSubscribers)) {
    try { subscriber.onChange(queue); } catch { /* ignore subscriber error */ }
  }
}

function deliverQueueError(cause: unknown) {
  for (const subscriber of Array.from(queueSubscribers)) {
    try { subscriber.onError?.(cause); } catch { /* ignore subscriber error */ }
  }
}

async function pollQueueOnce() {
  if (!queueSubscribers.size || document.hidden || queuePollInFlight) {
    scheduleQueuePoll();
    return;
  }
  clearQueuePollTimer();
  queuePollInFlight = true;
  const before = queueRevision;
  const controller = new AbortController();
  queuePollController = controller;
  try {
    const queue = await loraRequest("queue/status?monitor=1", undefined, controller.signal);
    if (!controller.signal.aborted && before === queueRevision) {
      deliverQueue(queue);
    }
  } catch (cause) {
    if (!controller.signal.aborted && before === queueRevision) {
      deliverQueueError(cause);
    }
  } finally {
    if (queuePollController === controller) {
      queuePollController = null;
    }
    queuePollInFlight = false;
    scheduleQueuePoll();
  }
}

function syncQueueTransport() {
  const needsWs = hasActiveQueueWorkerSubscriber();
  if (needsWs && !queueWsUnsubscribe) {
    queueWsUnsubscribe = subscribeWorkshopEvents(
      value => {
        if (value.type === "queue_updated" && value.data) {
          deliverQueue(value.data);
        } else if (
          value.type === "queue_progress"
          && value.data
          && latestQueue?.items?.some((item: LoraData) => item.id === value.data.item_id && item.status === "processing")
        ) {
          deliverQueue({
            ...latestQueue,
            items: latestQueue.items.map((item: LoraData) => (
              item.id === value.data.item_id && item.status === "processing"
                ? {
                    ...item,
                    progress: value.data.progress,
                    progress_detail: value.data.detail,
                    subtasks: value.data.subtasks ?? item.subtasks,
                  }
                : item
            )),
          });
        }
      },
      {
        onOpen: () => {
          // If the worker just started for this WebSocket, pull an initial snapshot right away.
          if (Date.now() - latestQueueUpdatedAt > 800 || !(latestQueue?.items?.length)) {
            void pollQueueOnce();
          } else {
            scheduleQueuePoll();
          }
        },
        onReconnect: () => {
          void pollQueueOnce();
        },
      },
    );
  } else if (!needsWs && queueWsUnsubscribe) {
    queueWsUnsubscribe();
    queueWsUnsubscribe = null;
    scheduleQueuePoll();
  }
}

export function pushQueueStatus(queue: LoraData) {
  deliverQueue(queue);
  scheduleQueuePoll();
}

export function refreshQueueStatus() {
  void pollQueueOnce();
}

export function subscribeQueueStatus(
  onChange: (queue: LoraData) => void,
  options?: { onError?: (cause: unknown) => void; activateWorker?: boolean },
): () => void {
  const subscriber: QueueSubscriber = {
    onChange,
    onError: options?.onError,
    activateWorker: Boolean(options?.activateWorker),
  };
  const wasEmpty = queueSubscribers.size === 0;
  const hadActiveWorker = hasActiveQueueWorkerSubscriber();
  queueSubscribers.add(subscriber);
  syncQueueTransport();
  if (latestQueue !== null) {
    try { onChange(latestQueue); } catch { /* ignore subscriber error */ }
  }
  if (wasEmpty || (!hadActiveWorker && subscriber.activateWorker) || Date.now() - latestQueueUpdatedAt > 2500) {
    void pollQueueOnce();
  } else {
    scheduleQueuePoll();
  }
  return () => {
    queueSubscribers.delete(subscriber);
    if (!queueSubscribers.size) {
      clearQueuePollTimer();
      queuePollController?.abort();
      queuePollController = null;
      queuePollInFlight = false;
    }
    syncQueueTransport();
  };
}

// ─── Shared LLM Streams Monitor Store ──────────────────────────────────

export type LlmStreamsMonitorState = {
  count: number | null;
  error: string;
  loading: boolean;
};

type LlmStreamsSubscriber = (state: LlmStreamsMonitorState) => void;

const llmStreamsSubscribers = new Set<LlmStreamsSubscriber>();
let llmStreamsState: LlmStreamsMonitorState = { count: null, error: "", loading: true };
let llmStreamsExternalFeeds = 0;
let llmStreamsPollTimer: number | null = null;
let llmStreamsPollController: AbortController | null = null;
let llmStreamsPollInFlight = false;

function clearLlmStreamsTimer() {
  if (llmStreamsPollTimer !== null) {
    window.clearTimeout(llmStreamsPollTimer);
    llmStreamsPollTimer = null;
  }
}

function deliverLlmStreamsState(next: LlmStreamsMonitorState) {
  llmStreamsState = next;
  for (const subscriber of Array.from(llmStreamsSubscribers)) {
    try { subscriber(next); } catch { /* ignore subscriber error */ }
  }
}

function scheduleLlmStreamsPoll() {
  clearLlmStreamsTimer();
  if (!llmStreamsSubscribers.size || llmStreamsExternalFeeds > 0 || document.hidden) return;
  llmStreamsPollTimer = window.setTimeout(() => {
    llmStreamsPollTimer = null;
    void pollLlmStreamsOnce();
  }, 3000);
}

async function pollLlmStreamsOnce() {
  if (!llmStreamsSubscribers.size || llmStreamsExternalFeeds > 0 || document.hidden || llmStreamsPollInFlight) {
    scheduleLlmStreamsPoll();
    return;
  }
  clearLlmStreamsTimer();
  llmStreamsPollInFlight = true;
  const controller = new AbortController();
  llmStreamsPollController = controller;
  try {
    const live = await llmLogsRequest<{ streams: LlmStream[]; errors: string[] }>("streams?monitor=1", controller.signal);
    if (controller.signal.aborted || llmStreamsExternalFeeds > 0) return;
    if (live.errors.length) {
      deliverLlmStreamsState({ count: null, error: live.errors.join(" "), loading: false });
    } else {
      deliverLlmStreamsState({ count: activeLlmStreams(live.streams).length, error: "", loading: false });
    }
  } catch (cause) {
    if (!controller.signal.aborted && llmStreamsExternalFeeds === 0) {
      deliverLlmStreamsState({
        count: null,
        error: cause instanceof Error ? cause.message : String(cause),
        loading: false,
      });
    }
  } finally {
    if (llmStreamsPollController === controller) {
      llmStreamsPollController = null;
    }
    llmStreamsPollInFlight = false;
    scheduleLlmStreamsPoll();
  }
}

export function claimLlmStreamsFeed(): () => void {
  llmStreamsExternalFeeds++;
  clearLlmStreamsTimer();
  llmStreamsPollController?.abort();
  llmStreamsPollController = null;
  return () => {
    llmStreamsExternalFeeds = Math.max(0, llmStreamsExternalFeeds - 1);
    if (llmStreamsExternalFeeds === 0 && llmStreamsSubscribers.size) {
      void pollLlmStreamsOnce();
    }
  };
}

export function publishLlmStreams(streams: LlmStream[] | null, error = "") {
  if (error) {
    deliverLlmStreamsState({ count: null, error, loading: false });
  } else if (streams) {
    deliverLlmStreamsState({ count: activeLlmStreams(streams).length, error: "", loading: false });
  }
}

export function subscribeLlmStreamsMonitor(subscriber: LlmStreamsSubscriber): () => void {
  const wasEmpty = llmStreamsSubscribers.size === 0;
  llmStreamsSubscribers.add(subscriber);
  subscriber(llmStreamsState);
  if (wasEmpty && llmStreamsExternalFeeds === 0) {
    void pollLlmStreamsOnce();
  } else {
    scheduleLlmStreamsPoll();
  }
  return () => {
    llmStreamsSubscribers.delete(subscriber);
    if (!llmStreamsSubscribers.size) {
      clearLlmStreamsTimer();
      llmStreamsPollController?.abort();
      llmStreamsPollController = null;
      llmStreamsPollInFlight = false;
    }
  };
}

// ─── Browser Tab Visibility Coordination ───────────────────────────────

if (typeof document !== "undefined") {
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      clearQueuePollTimer();
      clearLlmStreamsTimer();
      return;
    }
    if (eventSubscribers.size && (!eventSocket || eventSocket.readyState === WebSocket.CLOSED)) {
      connectEventSocket();
    }
    if (queueSubscribers.size) {
      void pollQueueOnce();
    }
    if (llmStreamsSubscribers.size && llmStreamsExternalFeeds === 0) {
      void pollLlmStreamsOnce();
    }
  });
}
