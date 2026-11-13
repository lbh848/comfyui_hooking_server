import { useEffect, useId, useRef, useState } from "react";
import { Activity, ChevronDown, ChevronUp, Copy, RefreshCw, ScrollText, Search } from "lucide-react";
import { Badge, Panel } from "./components";
import { llmSlotIds, llmSlotLabel } from "./llm";
import {
  actionLabels, activeLlmStreams, contentText, inputText, llmLogsRequest, statusLabels, useLlmLogsPoll,
  type LlmHistory, type LlmStats, type LlmStream, type StreamAction,
} from "./llmLogs";
import "./llm-logs.css";

type Notify = (message: string) => void;
const number = (value?: number | null, digits = 0) => value == null ? "—" : value.toLocaleString("ko-KR", { maximumFractionDigits: digits });
// 별칭이 설정된 LLM은 기술적인 모델명 대신 별칭을 표시한다.
const modelLabel = (entry: { llm_alias?: string; model?: string; service?: string }) =>
  entry.llm_alias || entry.model || entry.service || "모델 미기록";

function Stats({ stats }: { stats: LlmStats }) {
  return <div className="llm-log-stats">
    <span>입력 <b>{number(stats.prompt_tokens)}</b></span><span>출력(생각 포함) <b>{number(stats.completion_tokens)}</b> 토큰</span>
    <span title={stats.reasoning_tokens_estimated ? "수신한 생각 텍스트로 추정한 토큰 수입니다." : undefined}>그중 생각 <b>{stats.reasoning_tokens_estimated ? "약 " : ""}{number(stats.reasoning_tokens)}</b> 토큰</span>
    <span>경과 <b>{number(stats.elapsed, 1)}</b>초</span><span title="생각 토큰과 답변 토큰을 합한 전체 생성 토큰 수 ÷ 경과 시간"><b>{number(stats.tps, 1)}</b> 토큰/초 (생각 포함)</span>
    <span>첫 응답 <b>{number(stats.ttft, 2)}</b>초</span>
    {stats.tokens_estimated && <span title="정확한 사용량이 없는 경우 입력 텍스트와 수신한 생각·답변 텍스트로 토큰 수와 평균 속도를 추정합니다. 이미지·파일 분량은 제외됩니다.">토큰·속도 추정값</span>}
  </div>;
}
function Status({ status = "ok" }: { status?: string }) {
  return <Badge tone={status === "error" ? "rose" : ["ok", "race_won"].includes(status) ? "green" : "neutral"}>{statusLabels[status] || status}</Badge>;
}
function Output({ text, label, follow = false }: { text: string; label: string; follow?: boolean }) {
  const element = useRef<HTMLPreElement>(null);
  const atBottom = useRef(true);
  useEffect(() => {
    if (follow && atBottom.current && element.current) element.current.scrollTop = element.current.scrollHeight;
  }, [text, follow]);
  return <pre ref={element} className="llm-log-output" tabIndex={0} aria-label={label}
    onScroll={() => { const node = element.current; if (node) atBottom.current = node.scrollHeight - node.scrollTop - node.clientHeight < 32; }}>{text || "표시할 내용이 없습니다."}</pre>;
}
function Content({ title, text, notify }: { title: string; text: string; notify: Notify }) {
  return <section className="llm-log-content">
    <div className="llm-log-content-heading"><h4>{title}</h4><button className="icon-button" disabled={!text} title={`${title} 복사`} aria-label={`${title} 복사`} onClick={async () => {
      try { await navigator.clipboard.writeText(text); notify(`${title}을 복사했습니다.`); }
      catch { notify("복사하지 못했습니다. 내용을 선택해 복사하세요."); }
    }}><Copy size={14} /></button></div>
    <Output text={text} label={title} />
  </section>;
}
function HistoryDetail({ selected, notify }: { selected: LlmHistory | null; notify: Notify }) {
  const [record, setRecord] = useState<LlmHistory | null>(null);
  const [error, setError] = useState("");
  const [retry, setRetry] = useState(0);
  const identity = selected?.id;
  useEffect(() => {
    setRecord(null); setError("");
    if (!identity) return;
    const controller = new AbortController();
    void llmLogsRequest<{ record: LlmHistory }>(`history/${encodeURIComponent(identity)}`, controller.signal)
      .then((result) => { if (!controller.signal.aborted) setRecord(result.record); })
      .catch((error: Error) => { if (!controller.signal.aborted) setError(error.message); });
    return () => controller.abort();
  }, [identity, retry]);
  if (!selected) return <div className="llm-log-empty"><ScrollText size={24} /><strong>로그를 선택하세요</strong><span>입력과 출력 전체를 이곳에서 확인할 수 있습니다.</span></div>;
  const current = record?.id === identity ? record : null;
  return <div className="llm-log-detail" aria-label="선택한 로그 상세">
    <div className="llm-log-detail-heading"><h3>{selected.call_name || selected.task_key || "LLM 호출"}</h3><Status status={selected.status} /></div>
    <p className="llm-log-meta">{selected.ts?.replace("T", " ")} · {llmSlotLabel(selected.llm_slot)} · {modelLabel(selected)}</p>
    <Stats stats={current || selected} />
    {error ? <div className="llm-log-error" role="alert">{error}<button className="button secondary small" onClick={() => setRetry((value) => value + 1)}>다시 불러오기</button></div>
      : !current ? <p className="llm-log-message" role="status">로그 상세를 불러오는 중입니다…</p>
      : <>
        {current.error && <Content title="오류 내용" text={current.error} notify={notify} />}
        <Content title="입력 전체" text={inputText(current.input)} notify={notify} />
        {(current.reasoning_text || current.reasoning) && <Content title="추론 내용" text={current.reasoning_text || current.reasoning || ""} notify={notify} />}
        <Content title={current.status === "error" ? "LLM 원본 응답" : "출력 전체"} text={contentText(current.output)} notify={notify} />
        <details className="llm-log-metadata"><summary>전체 기록</summary><Content title="기록 JSON" text={contentText(current)} notify={notify} /></details>
      </>}
  </div>;
}
function LiveCard({ stream, streams, connected, follow, notify, refresh }: {
  stream: LlmStream; streams: LlmStream[]; connected: boolean; follow: boolean; notify: Notify; refresh: () => void;
}) {
  const [busy, setBusy] = useState(false);
  const [contentExpanded, setContentExpanded] = useState(false);
  const contentId = useId();
  const pending = useRef(false);
  const racePeer = !!stream.race_id && streams.some((peer) => peer.source === stream.source && peer.stream_id !== stream.stream_id && peer.race_id === stream.race_id);
  const parallelAvailable = stream.parallel_retry_available ?? stream.parallel_retry_supported;
  async function control(action: StreamAction) {
    if (pending.current) return;
    pending.current = true; setBusy(true);
    try {
      await llmLogsRequest(`streams/${stream.source}/${encodeURIComponent(stream.stream_id)}/control`, undefined, { action });
      notify(`${actionLabels[action]} 요청을 보냈습니다.`); refresh();
    } catch (error) { notify((error as Error).message); }
    finally { pending.current = false; setBusy(false); }
  }
  const elapsed = stream.started_at ? Math.max(stream.elapsed || 0, Date.now() / 1000 - stream.started_at) : stream.elapsed;
  const reasoning = stream.reasoning_text || stream.reasoning;
  return <article className="llm-live-card">
    <header><Badge tone="lavender">{llmSlotLabel(stream.llm_slot)}</Badge><strong>{stream.call_name || stream.task_key || "LLM 호출"}</strong><Badge tone={connected ? "green" : "neutral"}>{connected ? stream.status || "응답 중" : "연결 확인 중"}</Badge></header>
    <div className="llm-log-meta">{modelLabel(stream)}{stream.race_role && <span> · {stream.race_role === "original" ? "원본" : stream.race_role === "parallel" ? "병렬 재시도" : stream.race_role}</span>}</div>
    <div className="llm-live-controls">
      <button className="button secondary small" disabled={busy || !connected || !!stream.race_id} title="현재 연결을 끊고 같은 요청을 다시 실행" onClick={() => void control("retry")}>재시도</button>
      <button className="button secondary small" disabled={busy || !connected || !parallelAvailable || racePeer} title="같은 요청을 하나 더 실행하고 먼저 완료된 응답을 사용" onClick={() => void control("parallel_retry")}>병렬로 재시도</button>
      <button className="button secondary small" disabled={busy || !connected || !stream.text} title="현재까지 받은 내용을 결과로 사용" onClick={() => void control("use_partial")}>현재 내용 사용</button>
      <button className="button danger small" disabled={busy || !connected} onClick={() => void control("cancel")}>{busy ? "요청 중…" : "중지"}</button>
    </div>
    <Stats stats={{ ...stream, elapsed }} />
    <button type="button" className="button secondary small llm-live-content-toggle" aria-expanded={contentExpanded} aria-controls={contentId}
      onClick={() => setContentExpanded((value) => !value)}>
      {contentExpanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}{contentExpanded ? "내용 접기" : "내용 펼치기"}
    </button>
    <div id={contentId} className="llm-live-content" hidden={!contentExpanded}>
      {contentExpanded && <>
        {reasoning && <section><h4>추론 내용</h4><Output label="라이브 추론 내용" text={reasoning} follow={follow} /></section>}
        <section><h4>실시간 출력</h4><Output label="라이브 출력" text={stream.text || "첫 응답을 기다리고 있습니다…"} follow={follow} /></section>
      </>}
    </div>
  </article>;
}

export function LlmLogsPage({ notify }: { notify: Notify }) {
  const [revision, setRevision] = useState(0);
  const history = useLlmLogsPoll<{ history: LlmHistory[]; count: number }>("history", 3000, revision);
  const live = useLlmLogsPoll<{ streams: LlmStream[]; errors: string[] }>("streams", 750, revision);
  const [selected, setSelected] = useState<LlmHistory | null>(null);
  const [query, setQuery] = useState("");
  const [status, setStatus] = useState("");
  const [slot, setSlot] = useState("");
  const [follow, setFollow] = useState(true);
  const records = history.data?.history || [];
  const streams = activeLlmStreams(live.data?.streams);
  const hasLiveStreams = streams.length > 0;
  const search = query.trim().toLocaleLowerCase();
  const visible = records.filter((record) => (!status || (record.status || "ok") === status)
    && (!slot || record.llm_slot === slot)
    && (!search || [record.call_name, record.task_key, record.model, record.llm_alias, record.preview, record.ts].join(" ").toLocaleLowerCase().includes(search)));
  const connected = !!live.data && !live.error && !live.data.errors.length;
  const refresh = () => setRevision((value) => value + 1);

  return <div className="llm-logs-page">
    <div className="llm-logs-toolbar"><div><span className={`llm-live-indicator ${connected ? "connected" : ""}`} /><span role="status">{live.error || live.data?.errors.length ? "라이브 재연결 중" : live.data ? "자동 갱신 중" : "라이브 연결 중…"}</span><span className="llm-log-meta">로그 {records.length}건 · 라이브 {streams.length}개</span></div><button className="button secondary" onClick={refresh}><RefreshCw size={14} />새로고침</button></div>
    <div className="llm-logs-layout">
      <div className="llm-logs-sidebar">
        <Panel className="llm-history-panel" title={<><ScrollText size={16} />로그 이력</>} aside={<Badge>{records.length}건</Badge>}>
          <div className="llm-log-filters">
            <label className="llm-log-search"><Search size={15} /><input aria-label="LLM 로그 검색" placeholder="작업·모델·출력 요약 검색" value={query} onChange={(event) => setQuery(event.target.value)} /></label>
            <div><select aria-label="로그 상태" value={status} onChange={(event) => setStatus(event.target.value)}><option value="">모든 상태</option>{Object.entries(statusLabels).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select>
              <select aria-label="로그 LLM 슬롯" value={slot} onChange={(event) => setSlot(event.target.value)}><option value="">모든 LLM</option>{llmSlotIds.map((id) => <option key={id} value={id}>{llmSlotLabel(id)}</option>)}</select></div>
          </div>
          {history.error && <p className="llm-log-error" role="alert">{history.error}</p>}
          <div className="llm-history-list" aria-label="LLM 로그 목록" tabIndex={0}>
            {!history.data ? <p className="llm-log-message" role="status">{history.error ? "로그를 불러오지 못했습니다." : "로그를 불러오는 중입니다…"}</p>
              : !visible.length ? <p className="llm-log-message">{records.length ? "검색 조건에 맞는 로그가 없습니다." : "아직 기록된 LLM 호출이 없습니다."}</p>
              : visible.map((record) => <button key={record.id} className={`llm-history-row ${selected?.id === record.id ? "selected" : ""}`} aria-pressed={selected?.id === record.id} onClick={() => setSelected(record)}>
                <span className="llm-history-row-heading"><strong>{record.call_name || record.task_key || "LLM 호출"}</strong><Status status={record.status} /></span>
                <span className="llm-log-meta">{record.ts?.replace("T", " ")} · {llmSlotLabel(record.llm_slot)} · {modelLabel(record)}</span>
                <span className="llm-history-preview">{record.preview || "출력 내용 없음"}</span>
                <span className="llm-log-meta">입력 {number(record.prompt_tokens)} · 출력(생각 포함) {number(record.completion_tokens)} · 생각 {record.reasoning_tokens_estimated ? "약 " : ""}{number(record.reasoning_tokens)} 토큰</span>
                <span className="llm-log-meta">{number(record.elapsed, 1)}초 · {number(record.tps, 1)} 토큰/초 (생각 포함){record.tokens_estimated ? " · 추정값" : ""}</span>
              </button>)}
          </div>
        </Panel>
        <Panel className={`llm-live-panel ${hasLiveStreams ? "" : "is-collapsed"}`} title={<><Activity size={16} />라이브</>} aside={<div className="llm-live-options">
          <Badge tone={streams.length ? "green" : "neutral"}>{streams.length}개 진행 중</Badge>
          {hasLiveStreams && <button type="button" className="llm-auto-scroll" role="switch" aria-checked={follow}
            aria-label="자동 스크롤" title={follow ? "새 응답을 따라 스크롤합니다" : "현재 스크롤 위치를 유지합니다"}
            onClick={() => setFollow((value) => !value)}>
            <span>자동 스크롤</span>
            <span className="llm-auto-scroll-track" aria-hidden="true"><span /></span>
          </button>}
        </div>}>
          {(live.error || !!live.data?.errors.length) && <p className="llm-log-error" role="alert">{live.error || live.data?.errors.join(" ")}</p>}
          {hasLiveStreams && <div className="llm-live-list">
            {streams.map((stream) => <LiveCard key={`${stream.source}:${stream.stream_id}`} stream={stream} streams={streams} connected={!live.error && (stream.source === "settings" || !live.data?.errors.length)} follow={follow} notify={notify} refresh={refresh} />)}
          </div>}
        </Panel>
      </div>
      <Panel className="llm-detail-panel" title={<><ScrollText size={16} />로그 상세</>}>
        <div className="llm-history-detail-scroll"><HistoryDetail selected={selected} notify={notify} /></div>
      </Panel>
    </div>
  </div>;
}
