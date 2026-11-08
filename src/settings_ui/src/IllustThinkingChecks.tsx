import { useCallback, useEffect, useRef, useState } from "react";
import { Check, CircleHelp, LoaderCircle, TriangleAlert } from "lucide-react";
import { Empty, Panel } from "./components";
import { IllustrationFeedback, IllustrationLoading } from "./IllustBackups";
import { illustrationApi, illustrationRequest, type IllustrationData } from "./illustration";

type ThinkingStatus = "observed" | "zero" | "unknown" | "error" | "cancelled" | "checking" | "stale" | "idle";
type ThinkingResult = { status: ThinkingStatus; label: string; signature?: string; tokens?: number; chars?: number; elapsed?: number };
type ThinkingConnections = { routes: Record<string, { primary: string | null; fallback: string | null }>; slots: Record<string, { model: string; alias?: string; service: string; signature?: string }> };
type ThinkingTone = "green" | "yellow" | "red" | "gray";
type ThinkingProblem = { label: string; description: string };

export function illustrationNodeName(name: string) {
  return name.replaceAll("CALL2-SELECT", "SLOT-SELECTER").replaceAll("CALL2-PLAN", "PLAN")
    .replaceAll("DETAIL-COMPOSITE", "DETAIL 합성 삽화").replaceAll("CALL2-DETAIL", "DETAIL").replaceAll("CALL2-KEYVIS", "KEYVIS").replaceAll("CALL3", "POST PROCESS");
}

function thinkingAssignments(nodes: IllustrationData[], connections: ThinkingConnections | null) {
  const assignments = new Map<string, Set<string>>();
  for (const node of nodes.filter(item => item.enabled && item.task_key)) {
    const route = connections?.routes[node.task_key];
    for (const [role, slot] of [["메인", route?.primary], ["폴백", route?.fallback]]) {
      if (!slot || !connections?.slots[slot]) continue;
      if (!assignments.has(slot)) assignments.set(slot, new Set());
      assignments.get(slot)!.add(`${illustrationNodeName(node.name).replace(/\s+slot=.*/, "")} (${role})`);
    }
  }
  return new Map([...assignments].sort(([left], [right]) => left.localeCompare(right, "en", { numeric: true })));
}

export function useThinkingChecks(nodes: IllustrationData[], active: boolean, revision: number) {
  const [connections, setConnections] = useState<ThinkingConnections | null>(null);
  const [error, setError] = useState("");
  const [results, setResults] = useState<Record<string, ThinkingResult>>({});
  const [progress, setProgress] = useState<{ slot: string; completed: number; total: number } | null>(null);
  const controller = useRef<AbortController | null>(null);
  const nodesRef = useRef(nodes); nodesRef.current = nodes;
  const refresh = useCallback(async (signal: AbortSignal) => {
    const data = await illustrationRequest("illustration/thinking-connections", undefined, signal) as ThinkingConnections;
    if (!signal.aborted) { setConnections(data); setError(""); }
    return data;
  }, []);
  useEffect(() => {
    if (!active) return;
    const abort = new AbortController();
    const reload = () => { if (!controller.current) void refresh(abort.signal).catch(e => { if (!abort.signal.aborted) { setConnections(null); setError(e.message); } }); };
    reload(); window.addEventListener("focus", reload);
    return () => { abort.abort(); window.removeEventListener("focus", reload); };
  }, [active, revision, refresh]);
  useEffect(() => () => controller.current?.abort(), []);

  const assignments = thinkingAssignments(nodes, connections);
  function resultForSlot(slot: string | null | undefined): ThinkingResult {
    if (!slot) return { status: "idle", label: "설정 없음" };
    if (progress?.slot === slot) return { status: "checking", label: "대기·검사 중…" };
    const result = results[slot];
    if (!result) return { status: "idle", label: "미검사" };
    if (!connections?.slots[slot]?.signature || result.signature !== connections.slots[slot].signature) return { status: "stale", label: "연결 설정 변경 · 재검사 필요" };
    return result;
  }
  function slotLabel(slot: string) {
    const info = connections?.slots[slot];
    return `${slot.toUpperCase()} · ${info?.alias ? `${info.alias} · ` : ""}${info?.model || "모델 미설정"}`;
  }
  function nodeHealth(node: IllustrationData): { tone: ThinkingTone; label: string; problems: ThinkingProblem[] } | null {
    if (!node.enabled || !node.task_key || !connections) return null;
    const route = connections.routes[node.task_key];
    const main = resultForSlot(route?.primary), fallback = resultForSlot(route?.fallback);
    const problems: ThinkingProblem[] = [];
    if (!route?.primary) problems.push({ label: "메인 연결 미설정", description: "메인 연결이 설정되지 않았습니다." });
    else if (!connections.slots[route.primary]?.model?.trim()) problems.push({ label: "메인 모델 미설정", description: "메인 모델이 설정되지 않았습니다." });
    if (!route?.fallback) problems.push({ label: "폴백 미설정", description: "폴백이 설정되지 않았습니다." });
    else if (route.fallback === route.primary) problems.push({ label: "메인·폴백 동일", description: "폴백이 메인과 같은 연결입니다." });
    else if (!connections.slots[route.fallback]?.model?.trim()) problems.push({ label: "폴백 모델 미설정", description: "폴백 모델이 설정되지 않았습니다." });
    if (main.status === "observed") problems.push({ label: "메인 Think 사용", description: "메인 LLM에서 Think 사용이 확인되었습니다." });
    if (main.status === "error") problems.push({ label: "메인 검사 실패", description: "메인 LLM 검사 요청이 실패했습니다." });
    if (fallback.status === "error") problems.push({ label: "폴백 검사 실패", description: "폴백 LLM 검사 요청이 실패했습니다." });
    if (problems.length) return { tone: "red", label: problems.map(problem => problem.label).join(" · "), problems };
    if ([main, fallback].some(result => result.status === "checking")) return { tone: "gray", label: "검사 중…", problems };
    if ([main, fallback].some(result => result.status === "stale")) return { tone: "gray", label: "재검사 필요", problems };
    if (fallback.status === "observed") return { tone: "yellow", label: "폴백 Think 사용", problems };
    if ([main, fallback].every(result => ["zero", "unknown"].includes(result.status))) return { tone: "green", label: "Think 미사용", problems };
    return { tone: "gray", label: "검사 대기", problems };
  }

  async function check(target?: string) {
    if (controller.current) return;
    const abort = new AbortController(); controller.current = abort;
    setProgress({ slot: "", completed: 0, total: 0 }); setError("");
    try {
      const saved = await refresh(abort.signal);
      const currentAssignments = thinkingAssignments(nodesRef.current, saved);
      const slots = target ? [target].filter(slot => currentAssignments.has(slot)) : [...currentAssignments.keys()];
      if (!slots.length) throw new Error("검사할 LLM 연결이 없습니다. 현재 흐름과 연결 설정을 확인하세요.");
      for (const [index, slot] of slots.entries()) {
        if (abort.signal.aborted) break;
        setProgress({ slot, completed: index, total: slots.length });
        const signature = saved.slots[slot]?.signature;
        try {
          if (!signature) throw new Error("연결 설정을 확인하려면 설정 서버를 다시 시작하세요.");
          if (!saved.slots[slot].model.trim()) throw new Error("모델이 설정되지 않았습니다.");
          const response = await fetch(`${illustrationApi}/illustration/thinking-check`, {
            method: "POST", signal: abort.signal, cache: "no-store",
            headers: { "Content-Type": "application/json", "X-Comfy-Settings": "1", "X-Comfy-Settings-Schema": "2" },
            body: JSON.stringify({ target: slot, thinking_check: true, signature }),
          });
          if (!response.ok || !response.body) throw new Error((await response.json().catch(() => null))?.error || `검사 요청 실패 (${response.status})`);
          const reader = response.body.getReader(), decoder = new TextDecoder();
          let buffer = "", done: IllustrationData | null = null;
          const dispatch = (frame: string) => {
            const lines = frame.split("\n"), type = lines.find(line => line.startsWith("event:"))?.slice(6).trim();
            const raw = lines.filter(line => line.startsWith("data:")).map(line => line.slice(5).trimStart()).join("\n");
            if (!raw) return;
            const value = JSON.parse(raw);
            if (type === "error" || type === "cancelled") throw new Error(value.error || value.reason || "검사를 중지했습니다.");
            if (type === "done") done = value;
          };
          try {
            while (true) {
              const chunk = await reader.read();
              buffer += decoder.decode(chunk.value, { stream: !chunk.done }); buffer = buffer.replaceAll("\r\n", "\n");
              let end;
              while ((end = buffer.indexOf("\n\n")) >= 0) { dispatch(buffer.slice(0, end)); buffer = buffer.slice(end + 2); }
              if (chunk.done) { if (buffer.trim()) dispatch(buffer); break; }
            }
          } finally { await reader.cancel().catch(() => {}); reader.releaseLock(); }
          if (!done) throw new Error("검사 결과 없이 연결이 종료되었습니다.");
          const result = done as IllustrationData;
          const status = result.thinking_status === "observed" ? "observed" : result.thinking_status === "zero" ? "zero" : "unknown";
          const label = status === "observed" ? "Think 사용 확인" : status === "zero" ? "이번 요청 추론 0토큰" : "Think 미사용 · 추론 정보 미수신";
          setResults(current => ({ ...current, [slot]: { signature, status, label, tokens: result.reasoning_tokens, chars: result.reasoning_chars, elapsed: result.elapsed } }));
        } catch (e) {
          setResults(current => ({ ...current, [slot]: { signature, status: abort.signal.aborted ? "cancelled" : "error", label: abort.signal.aborted ? "검사를 중지했습니다." : (e as Error).message } }));
        }
      }
      if (!abort.signal.aborted) await refresh(abort.signal);
    } catch (e) { if (!abort.signal.aborted) { setConnections(null); setError((e as Error).message); } }
    finally { controller.current = null; setProgress(null); }
  }
  return { connections, error, assignments, progress, resultForSlot, slotLabel, nodeHealth, check, stop: () => controller.current?.abort() };
}

export type ThinkingChecksState = ReturnType<typeof useThinkingChecks>;

export function ThinkingBadge({ tone, label }: { tone: ThinkingTone; label: string }) {
  const Icon = tone === "green" ? Check : tone === "gray" ? CircleHelp : TriangleAlert;
  return <span className="illust-thinking-badge" data-tone={tone}><Icon size={12} aria-hidden="true" />{label}</span>;
}

function ThinkingResultText({ result }: { result: ThinkingResult }) {
  return <span>{result.label}{result.tokens != null ? ` · 추론 ${result.tokens}토큰` : (result.chars || 0) > 0 ? " · 추론 데이터 수신" : ""}{typeof result.elapsed === "number" && Number.isFinite(result.elapsed) && ` · ${result.elapsed.toFixed(2)}초`}</span>;
}

export function ThinkingNodeDetails({ node, thinking, disabled }: { node: IllustrationData; thinking: ThinkingChecksState; disabled: boolean }) {
  const health = thinking.nodeHealth(node), route = thinking.connections?.routes[node.task_key];
  if (!health) return null;
  return <section className="illust-thinking-node-details" aria-label={`${illustrationNodeName(node.name)} Think 확인`}>
    <ThinkingBadge {...health} />
    {!!health.problems.length && <ul className="field-error">{health.problems.map(problem => <li key={problem.label}>{problem.description}</li>)}</ul>}
    <div className="illust-thinking-connections">{([['메인', route?.primary], ['폴백', route?.fallback]] as const).map(([role, slot]) => <article key={role}>
      <strong>{role}</strong><p>{slot ? thinking.slotLabel(slot) : "설정 없음"}</p>
      {slot && <><p className="field-help" role="status"><ThinkingResultText result={thinking.resultForSlot(slot)} /></p><button type="button" className="button" disabled={disabled || !!thinking.progress} onClick={() => void thinking.check(slot)}>Think 확인</button></>}
    </article>)}</div>
  </section>;
}

export function ThinkingChecks({ thinking, disabled, onReload }: { thinking: ThinkingChecksState; disabled: boolean; onReload: () => void }) {
  const { connections, assignments, progress, error } = thinking;
  const completed = [...assignments.keys()].filter(slot => ["observed", "zero", "unknown", "error"].includes(thinking.resultForSlot(slot).status)).length;
  return <Panel title="삽화 LLM Think 확인" aside={<button type="button" className="button primary" disabled={disabled || !!progress || !assignments.size} onClick={() => void thinking.check()}>{progress ? "확인 중…" : "모두 확인"}</button>}>
    <div className="illust-body illust-form">
      <p className="field-help">현재 저장된 연결로 짧은 계산 요청을 보냅니다. API 사용량이 발생합니다. 같은 연결은 한 번만 검사합니다.</p>
      <p className="field-help">추론 블록이나 추론 토큰이 없는 응답은 Think 미사용으로 표시합니다. 검사 결과는 이번 요청의 응답 기준입니다.</p>
      <div className="illust-thinking-legend"><ThinkingBadge tone="green" label="메인·폴백 Think 미사용" /><ThinkingBadge tone="yellow" label="폴백 Think 사용" /><ThinkingBadge tone="red" label="메인 Think·연결 문제" /><ThinkingBadge tone="gray" label="검사 대기·재검사" /></div>
      {progress && <div className="illust-thinking-progress" role="status"><LoaderCircle size={15} aria-hidden="true" /><span>{progress.total ? `${progress.completed + 1}/${progress.total} · ${progress.slot.toUpperCase()} 대기·검사 중…` : "저장된 연결 확인 중…"}</span><button type="button" className="button danger" onClick={thinking.stop}>검사 중지</button></div>}
      {!connections ? <IllustrationLoading error={error} retry={onReload} /> : !assignments.size ? <Empty title="검사할 LLM 연결이 없습니다">현재 입력 방식과 생성 설정, 외부 LLM 분기를 확인하세요.</Empty> : <>
        <p className="field-help" role="status">{assignments.size}개 연결 · 검사 완료 {completed}개</p>
        <div className="illust-thinking-slots">{[...assignments].map(([slot, tasks]) => {
          const result = thinking.resultForSlot(slot);
          const tone: ThinkingTone = result.status === "error" ? "red" : result.status === "observed" ? "yellow" : ["zero", "unknown"].includes(result.status) ? "green" : "gray";
          return <article className="illust-thinking-slot" key={slot} data-thinking-slot={slot}>
            <header><div><h3>{thinking.slotLabel(slot)}</h3><span className="field-help">{connections.slots[slot].service}</span></div><button type="button" className="button" disabled={disabled || !!progress} onClick={() => void thinking.check(slot)}>Think 확인</button></header>
            <p className="illust-thinking-result" data-tone={tone} role="status"><ThinkingResultText result={result} /></p>
            <div className="illust-thinking-tasks">{[...tasks].map(task => <span key={task}>{task}</span>)}</div>
          </article>;
        })}</div>
      </>}
      {connections && <IllustrationFeedback error={error} />}
    </div>
  </Panel>;
}
