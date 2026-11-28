import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { Flame } from "lucide-react";
import { illustrationNodeName, illustrationRequest, type IllustrationData } from "./illustration";
import { useOptionalWorkshopScreen } from "./WorkshopNavigation";
import { subscribeWorkshopEvents } from "./workshopEvents";

export const flowStatusLabels: Record<string, string> = { waiting: "대기", processing: "처리 중", cancelling: "중단 중", completed: "완료", failed: "실패", cancelled: "취소", skipped: "생략" };

export interface IllustrationFlowHistoryEntry {
  id: string;
  label: string;
  group_label: string;
  status: string;
  created_at: number;
  updated_at: number;
  revision: number;
  active: boolean;
  backup_count: number;
}

// Fetch sequentially; keep the last result on reconnect, but never across resources.
export function useFlowPoll(path: string, { poll = true, refreshKey = "", enabled = true, interval = 2000 } = {}) {
  const [loaded, setLoaded] = useState<{ path: string; data: IllustrationData; signature: string } | null>(null);
  const [error, setError] = useState("");
  const [revision, setRevision] = useState(0);
  const intervalRef = useRef(interval);
  intervalRef.current = interval;
  const reload = useCallback(() => setRevision(value => value + 1), []);
  useEffect(() => {
    if (!enabled) return;
    const controller = new AbortController();
    let timer: number | undefined;
    let inFlight = false;
    let lastFailed = false;
    setError("");
    function schedule() {
      window.clearTimeout(timer);
      if (controller.signal.aborted || document.hidden) return;
      if (poll || lastFailed) {
        timer = window.setTimeout(() => { void load(); }, lastFailed ? 2500 : intervalRef.current);
      }
    }
    async function load() {
      if (controller.signal.aborted || document.hidden || inFlight) {
        schedule();
        return;
      }
      window.clearTimeout(timer);
      inFlight = true;
      lastFailed = false;
      try {
        const data = await illustrationRequest(path, undefined, controller.signal);
        // Preserve routing identifiers while displaying the current name in active and saved runs.
        const displayNode = (node: IllustrationData) => node?.label?.includes("EMOTION-RESOLVE")
          ? { ...node, label: illustrationNodeName(node.label) } : node;
        if (Array.isArray(data.flow?.nodes)) data.flow = { ...data.flow, nodes: data.flow.nodes.map(displayNode) };
        if (data.node) data.node = displayNode(data.node);
        if (!controller.signal.aborted) {
          const signature = JSON.stringify(data);
          setLoaded(current => current?.path === path && current.signature === signature ? current : { path, data, signature });
          setError("");
        }
      } catch (cause) {
        lastFailed = true;
        if (!controller.signal.aborted) setError((cause as Error).message);
      } finally {
        inFlight = false;
        schedule();
      }
    }
    const onVisibility = () => {
      if (document.hidden) window.clearTimeout(timer);
      else if (poll || lastFailed) void load();
    };
    document.addEventListener("visibilitychange", onVisibility);
    void load();
    return () => {
      controller.abort();
      window.clearTimeout(timer);
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, [path, revision, poll, refreshKey, enabled]);
  const data = loaded?.path === path ? loaded.data : null;
  return useMemo(() => ({ data, error, reload }), [data, error, reload]);
}

const HistoryContext = createContext<ReturnType<typeof useFlowPoll> | null>(null);

export function IllustrationFlowHistoryProvider({ children, scope = "illust" }: { children: ReactNode; scope?: "illust" | "emotion" }) {
  const screen = useOptionalWorkshopScreen();
  const active = screen ? screen.active : true;
  const [hasActiveFlow, setHasActiveFlow] = useState(false);
  // 삽화/감정 삽화 백업이 서로의 흐름 그룹을 참조하지 않도록 범위를 전달한다.
  const resource = useFlowPoll(`illustration_flow?history=1&scope=${scope}`, {
    enabled: active,
    interval: hasActiveFlow ? 2000 : 6000,
  });
  const activeCount = Number(resource.data?.active_count || 0);
  useEffect(() => { setHasActiveFlow(activeCount > 0); }, [activeCount]);
  useEffect(() => {
    if (!active) return;
    let debounceTimer: number | undefined;
    const unsubscribe = subscribeWorkshopEvents(value => {
      if (value.type === "queue_updated" || value.type === "backup_created") {
        window.clearTimeout(debounceTimer);
        debounceTimer = window.setTimeout(() => {
          if (!document.hidden) resource.reload();
        }, 300);
      }
    });
    return () => {
      window.clearTimeout(debounceTimer);
      unsubscribe();
    };
  }, [active, resource.reload]);
  return <HistoryContext.Provider value={resource}>{children}</HistoryContext.Provider>;
}

export function useIllustrationFlowHistory() {
  const resource = useContext(HistoryContext)!;
  return { ...resource, history: (resource.data?.history || []) as IllustrationFlowHistoryEntry[] };
}

export function IllustrationFlowActivity() {
  const { data, error } = useIllustrationFlowHistory();
  const count = error ? 0 : Number(data?.active_count || 0);
  return <span className="illust-flow-activity" role="status" aria-live="polite">
    {count > 0 && <span title={`삽화 작업 중 · ${count}개 흐름`} aria-label={`삽화 작업 중 · ${count}개 흐름`}><Flame size={17} aria-hidden="true" /></span>}
  </span>;
}
