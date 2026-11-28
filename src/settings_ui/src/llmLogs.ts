import { useEffect, useRef, useState } from "react";

export type LlmStats = {
  prompt_tokens?: number | null; completion_tokens?: number | null;
  pdf_tokens_before?: number | null;
  elapsed?: number | null; tps?: number | null; ttft?: number | null;
  tokens_estimated?: boolean;
  reasoning_tokens_estimated?: boolean;
  reasoning_tokens?: number | null; reasoning_chars?: number | null;
};
export type LlmHistory = LlmStats & {
  id: string; ts?: string; call_name?: string; task_key?: string;
  model?: string; llm_alias?: string; service?: string; status?: string; llm_slot?: string;
  preview?: string; input?: unknown; output?: unknown; error?: string;
  reasoning?: string; reasoning_text?: string; race_id?: string; race_role?: string; race_status?: string;
  [key: string]: unknown;
};
export type LlmStream = LlmStats & {
  stream_id: string; source: "settings" | "internal-lora"; active: boolean;
  call_name?: string; task_key?: string; llm_slot?: string; model?: string; llm_alias?: string;
  text?: string; reasoning?: string; reasoning_text?: string; status?: string; started_at?: number;
  race_id?: string; race_role?: string;
  parallel_retry_supported?: boolean; parallel_retry_available?: boolean;
};
export type StreamAction = "cancel" | "retry" | "parallel_retry" | "use_partial";
export function activeLlmStreams(streams: LlmStream[] = []): LlmStream[] {
  return streams.filter((stream) => stream.active);
}
export const actionLabels: Record<StreamAction, string> = {
  retry: "재시도", parallel_retry: "병렬로 재시도", use_partial: "현재 내용 사용", cancel: "중지",
};
export const statusLabels: Record<string, string> = {
  ok: "완료", error: "오류", cancelled: "취소", race_won: "채택", race_lost: "폐기",
};
export function contentText(value: unknown): string {
  return typeof value === "string" ? value : value == null ? "" : JSON.stringify(value, null, 2);
}
export function inputText(value: unknown): string {
  if (!Array.isArray(value)) return contentText(value);
  return value.map((message, index) => message && typeof message === "object" && "content" in message
    ? `[${index + 1}. ${String(message.role || "?").toUpperCase()}]\n${contentText(message.content)}`
    : contentText(message)).join("\n\n");
}
export async function llmLogsRequest<T>(path: string, signal?: AbortSignal, body?: unknown): Promise<T> {
  const response = await fetch(`/api/setting/llm/${path}`, {
    method: body === undefined ? "GET" : "POST", signal, cache: "no-store",
    headers: { "X-Comfy-Settings": "1", "X-Comfy-Settings-Schema": "2", ...(body === undefined ? {} : { "Content-Type": "application/json" }) },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const result = await response.json().catch(() => null);
  if (!response.ok || !result || result.success === false) {
    throw new Error(result?.error || (response.status === 404 ? "LLM 로그를 연결하려면 설정 서버를 다시 시작하세요." : `LLM 로그 요청에 실패했습니다. (${response.status})`));
  }
  return result as T;
}
export function useLlmLogsPoll<T>(path: string, interval: number, refresh: number, active = true) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState("");
  const intervalRef = useRef(interval);
  intervalRef.current = interval;
  useEffect(() => {
    if (!active) return;
    const controller = new AbortController();
    let timer: number | undefined;
    let inFlight = false;
    function schedule() {
      window.clearTimeout(timer);
      if (controller.signal.aborted || document.hidden) return;
      timer = window.setTimeout(() => { void poll(); }, intervalRef.current);
    }
    async function poll() {
      if (controller.signal.aborted || document.hidden || inFlight) {
        schedule();
        return;
      }
      window.clearTimeout(timer);
      inFlight = true;
      try {
        const result = await llmLogsRequest<T>(path, controller.signal);
        if (!controller.signal.aborted) { setData(result); setError(""); }
      } catch (error) {
        if (!controller.signal.aborted) setError((error as Error).message);
      } finally {
        inFlight = false;
        schedule();
      }
    }
    const onVisibility = () => {
      if (document.hidden) window.clearTimeout(timer);
      else void poll();
    };
    document.addEventListener("visibilitychange", onVisibility);
    void poll();
    return () => {
      controller.abort();
      window.clearTimeout(timer);
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, [path, refresh, active]);
  return { data, error };
}
