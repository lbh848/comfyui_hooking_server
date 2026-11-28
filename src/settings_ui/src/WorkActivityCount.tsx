import { useEffect, useState } from "react";
import type { LoraData } from "./internalLora";
import { isFinishedQueueItem } from "./workQueue";
import { subscribeLlmStreamsMonitor, subscribeQueueStatus } from "./workshopEvents";

export function WorkActivityCount({ page }: { page: "work-queue" | "llm-logs" }) {
  const [count, setCount] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const label = page === "work-queue" ? "작업" : "실행";

  useEffect(() => {
    if (page === "work-queue") {
      return subscribeQueueStatus(
        queue => {
          const items: LoraData[] = Array.isArray(queue.items) ? queue.items : [];
          // pending_count는 순수 대기 항목만 세어 실행 중 작업이 0으로 보이므로
          // 작업 큐 페이지와 같은 기준(종료 전 항목 = 실행 중 + 대기)으로 센다.
          setCount(items.filter(item => !isFinishedQueueItem(item)).length);
          setError("");
          setLoading(false);
        },
        {
          activateWorker: false,
          onError: cause => {
            setCount(null);
            setError(cause instanceof Error ? cause.message : String(cause));
            setLoading(false);
          },
        },
      );
    }
    return subscribeLlmStreamsMonitor(state => {
      setCount(state.count);
      setError(state.error);
      setLoading(state.loading);
    });
  }, [page]);

  const description = loading ? `${label} 개수 확인 중` : error
    ? `${label} 개수 확인 실패: ${error}`
    : page === "work-queue" ? `진행 중인 작업 ${count}개` : `동시에 실행 중인 LLM ${count}개`;
  return <small className={`nav-activity-count ${(count || 0) > 0 ? "is-busy" : ""}`}
    title={description} aria-label={description} aria-live="polite">
    {label} {loading ? "…" : count == null ? "—" : count.toLocaleString("ko-KR")}
  </small>;
}
