import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from "react";
import { Flame } from "lucide-react";
import { illustrationNodeName, illustrationRequest, type IllustrationData } from "./illustration";

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
export function useFlowPoll(path: string) {
  const [loaded, setLoaded] = useState<{ path: string; data: IllustrationData } | null>(null);
  const [error, setError] = useState("");
  const [revision, setRevision] = useState(0);
  const reload = useCallback(() => setRevision(value => value + 1), []);
  useEffect(() => {
    const controller = new AbortController();
    let timer: number;
    setError("");
    async function poll() {
      try {
        const data = await illustrationRequest(path, undefined, controller.signal);
        // Preserve routing identifiers while displaying the current name in active and saved runs.
        const displayNode = (node: IllustrationData) => node?.label?.includes("EMOTION-RESOLVE")
          ? { ...node, label: illustrationNodeName(node.label) } : node;
        if (Array.isArray(data.flow?.nodes)) data.flow = { ...data.flow, nodes: data.flow.nodes.map(displayNode) };
        if (data.node) data.node = displayNode(data.node);
        if (!controller.signal.aborted) { setLoaded({ path, data }); setError(""); }
      } catch (cause) {
        if (!controller.signal.aborted) setError((cause as Error).message);
      } finally {
        if (!controller.signal.aborted) timer = window.setTimeout(poll, 2000);
      }
    }
    void poll();
    return () => { controller.abort(); window.clearTimeout(timer); };
  }, [path, revision]);
  return { data: loaded?.path === path ? loaded.data : null, error, reload };
}

const HistoryContext = createContext<ReturnType<typeof useFlowPoll> | null>(null);

export function IllustrationFlowHistoryProvider({ children, scope = "illust" }: { children: ReactNode; scope?: "illust" | "emotion" }) {
  // 삽화/감정 삽화 백업이 서로의 흐름 그룹을 참조하지 않도록 범위를 전달한다.
  const resource = useFlowPoll(`illustration_flow?history=1&scope=${scope}`);
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
