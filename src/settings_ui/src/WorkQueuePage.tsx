import { useEffect, useState } from "react";
import { Badge, Panel } from "./components";
import { loraRequest, type LoraData } from "./internalLora";
import { illustrationRequest } from "./illustration";
import { isFinishedQueueItem, queueHistoryItems } from "./workQueue";
import { pushQueueStatus, subscribeQueueStatus } from "./workshopEvents";
import "./work-queue.css";

const statusLabels: Record<string, string> = {
  pending: "대기", processing: "실행 중", waiting: "대기",
  completed: "완료", failed: "실패", cancelled: "취소",
};

export function WorkQueuePage({ notify }: { notify: (message: string) => void }) {
  const [state, setState] = useState<LoraData | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [collapsedErrors, setCollapsedErrors] = useState<Set<string>>(() => new Set());

  useEffect(() => subscribeQueueStatus(
    result => {
      setState(result);
      setError("");
    },
    {
      activateWorker: true,
      onError: e => setError(e instanceof Error ? e.message : String(e)),
    },
  ), []);

  async function action(path: string, body: LoraData, item?: LoraData) {
    setBusy(true);
    try {
      if (path === "cancel" && item?.type === "visual_profile_guide") {
        await illustrationRequest("bot_mode/character_cards/suggest_metadata/cancel", { item_id: item.id });
      } else {
        await loraRequest(`queue/${path}`, body);
      }
      const next = await loraRequest("queue/status");
      pushQueueStatus(next);
    } catch (e) {
      notify((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  const items: LoraData[] = state?.items || [];
  const active = items.filter(item => !isFinishedQueueItem(item));
  const history = queueHistoryItems(items);
  const renderItem = (item: LoraData) => {
    const finished = isFinishedQueueItem(item);
    return <div key={item.id} className={`work-queue-row${item.status === "failed" ? " failed" : ""}`}>
      <div>
        <strong>{item.label}</strong>
        {item.status === "failed" ? <details className="work-queue-error" open={!collapsedErrors.has(item.id)} onToggle={event => {
          const open = event.currentTarget.open;
          setCollapsedErrors(current => {
            if (current.has(item.id) === !open) return current;
            const next = new Set(current);
            if (open) next.delete(item.id); else next.add(item.id);
            return next;
          });
        }}>
          <summary>실패 이유</summary>
          <pre tabIndex={0} aria-label={`${item.label} 실패 이유`}>{item.error || "실패 이유가 전달되지 않았습니다."}</pre>
        </details> : <p>{item.error || item.progress_detail?.message || ""}</p>}
        {!finished && item.status === "processing" && <progress aria-label={`${item.label} 진행률`} max={100} value={item.progress || 0} />}
      </div>
      <Badge tone={item.status === "failed" ? "rose" : item.status === "completed" ? "green" : "neutral"}>{statusLabels[item.status] || item.status}</Badge>
      <button className="button" disabled={busy} onClick={() => void action(finished ? "remove" : "cancel", { id: item.id }, item)}>{finished ? "목록에서 제거" : "취소"}</button>
    </div>;
  };

  return <Panel title="작업 큐" aside={<div className="work-queue-actions">
    <button className="button" disabled={busy || !state} onClick={() => void action("pause", { paused: !state?.paused })}>{state?.paused ? "재개" : "일시 정지"}</button>
    <button className="button" disabled={busy || !state?.pending_count} onClick={() => void action("cancel_all", {})}>대기 작업 취소</button>
  </div>}>
    <div className="work-queue-content">
      {error && <p className="field-error" role="alert">{error}</p>}
      {!state ? <p role="status">작업 큐를 연결하는 중입니다…</p>
        : <>
          <section aria-label="최근 작업">
            <h3 className="work-queue-heading">최근 작업 <span>최대 10개 · 최신순</span></h3>
            {history.length ? history.map(renderItem) : <p className="field-help">아직 작업 기록이 없습니다.</p>}
          </section>
          <section aria-label="진행 중인 작업">
            <h3 className="work-queue-heading">진행 중인 작업 <span>{active.length}개</span></h3>
            {active.length ? active.map(renderItem) : <p className="field-help">대기 중인 작업이 없습니다.</p>}
          </section>
        </>}
    </div>
  </Panel>;
}
