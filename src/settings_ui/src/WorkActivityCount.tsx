import { useEffect, useState } from "react";
import { loraRequest, type LoraData } from "./internalLora";
import { activeLlmStreams, llmLogsRequest, type LlmStream } from "./llmLogs";
import { isFinishedQueueItem } from "./workQueue";

export function WorkActivityCount({ page }: { page: "work-queue" | "llm-logs" }) {
  const [count, setCount] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const label = page === "work-queue" ? "작업" : "실행";

  useEffect(() => {
    const controller = new AbortController();
    let timer: number;
    async function poll() {
      try {
        let nextCount: number;
        if (page === "work-queue") {
          const queue = await loraRequest("queue/status?monitor=1", undefined, controller.signal);
          const items: LoraData[] = Array.isArray(queue.items) ? queue.items : [];
          // pending_count는 순수 대기 항목만 세어 실행 중 작업이 0으로 보이므로
          // 작업 큐 페이지와 같은 기준(종료 전 항목 = 실행 중 + 대기)으로 센다.
          nextCount = items.filter(item => !isFinishedQueueItem(item)).length;
        } else {
          const live = await llmLogsRequest<{ streams: LlmStream[]; errors: string[] }>("streams?monitor=1", controller.signal);
          if (live.errors.length) throw new Error(live.errors.join(" "));
          nextCount = activeLlmStreams(live.streams).length;
        }
        if (!controller.signal.aborted) { setCount(nextCount); setError(""); }
      } catch (error) {
        if (!controller.signal.aborted) { setCount(null); setError((error as Error).message); }
      } finally {
        if (!controller.signal.aborted) {
          setLoading(false);
          timer = window.setTimeout(poll, 2000);
        }
      }
    }
    void poll();
    return () => { controller.abort(); window.clearTimeout(timer); };
  }, [page]);

  const description = loading ? `${label} 개수 확인 중` : error
    ? `${label} 개수 확인 실패: ${error}`
    : page === "work-queue" ? `진행 중인 작업 ${count}개` : `동시에 실행 중인 LLM ${count}개`;
  return <small className={`nav-activity-count ${(count || 0) > 0 ? "is-busy" : ""}`}
    title={description} aria-label={description} aria-live="polite">
    {label} {loading ? "…" : count == null ? "—" : count.toLocaleString("ko-KR")}
  </small>;
}
