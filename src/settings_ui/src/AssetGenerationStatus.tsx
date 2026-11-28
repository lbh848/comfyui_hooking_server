import { useEffect, useState, type CSSProperties } from "react";
import { CircleCheck, CircleX, ListTodo } from "lucide-react";
import { assetGenerationStatus, matchesAssetQueueWorkspace, type AssetQueueState, type AssetQueueWorkspace } from "./assetGenerationQueue";
import { subscribeQueueStatus } from "./workshopEvents";

export function AssetGenerationStatus({ onOpenQueue, onStateChange, workspace = "asset" }: {
  onOpenQueue: () => void;
  onStateChange?: (state: AssetQueueState | null) => void;
  workspace?: AssetQueueWorkspace;
}) {
  const [state, setState] = useState<AssetQueueState | null>(null);
  const [error, setError] = useState("");
  const [result, setResult] = useState<{ failed: boolean; title: string } | null>(null);

  useEffect(() => {
    let resultTimer: number | undefined;
    let latest: AssetQueueState | null = null;
    const unsubscribe = subscribeQueueStatus(
      raw => {
        const next = raw as AssetQueueState;
        // The first snapshot is history, not a newly completed generation.
        if (latest) {
          const previous = new Map((latest.items || []).map(item => [item.id, item.status]));
          const finished = (next.items || []).filter(item => item.type === "asset_generation"
            && matchesAssetQueueWorkspace(item, workspace)
            && (item.status === "completed" || item.status === "failed") && previous.get(item.id) !== item.status)
            .sort((left, right) => (right.completed_at || 0) - (left.completed_at || 0))[0];
          if (finished) {
            setResult({ failed: finished.status === "failed", title: [finished.label, finished.error].filter(Boolean).join("\n") });
            window.clearTimeout(resultTimer);
            resultTimer = window.setTimeout(() => setResult(null), 6000);
          }
        }
        latest = next;
        setState(next);
        setError("");
      },
      {
        activateWorker: true,
        onError: cause => setError(cause instanceof Error ? cause.message : String(cause)),
      },
    );
    return () => { window.clearTimeout(resultTimer); unsubscribe(); };
  }, [workspace]);

  useEffect(() => { onStateChange?.(error ? null : state); }, [state, error, onStateChange]);

  const status = state ? assetGenerationStatus(state, workspace) : null;
  const available = status && !error;
  return <div className={`asset-generation-status${error ? " has-error" : ""}`} aria-label={workspace === "bg_asset" ? "배경 생성 현황" : "에셋 생성 현황"}>
    <div className="asset-generation-status-line" role="status">
      <span className="asset-generation-pending" title={error || undefined}>대기 <strong>{available ? status.pending.toLocaleString("ko-KR") : "—"}</strong></span>
      {available && status.label && <span className="asset-generation-status-label" role="progressbar" aria-label="현재 그리기 단계 진행률" aria-valuemin={0} aria-valuemax={100} aria-valuenow={status.percent ?? undefined} title={[status.title, status.detail].filter(Boolean).join("\n")} style={{ "--asset-progress": `${status.percent ?? 0}%` } as CSSProperties}>
        <span className="asset-generation-status-ring" aria-hidden="true">· </span>
        <span className="asset-generation-status-text">{status.label}</span>
      </span>}
      {available && status.paused && status.pending > 0 && <span className="asset-generation-paused">일시정지</span>}
      {result && <span className={`asset-generation-result${result.failed ? " is-failed" : ""}`} title={result.title || undefined}>
        {result.failed ? <CircleX size={14} aria-hidden="true" /> : <CircleCheck size={14} aria-hidden="true" />}
        {result.failed ? "생성 실패" : "생성 성공"}
      </span>}
    </div>
    <button type="button" className="icon-button asset-generation-queue" aria-label="작업 큐로 이동" title="작업 큐로 이동" onClick={onOpenQueue}><ListTodo size={16} aria-hidden="true" /></button>
  </div>;
}
