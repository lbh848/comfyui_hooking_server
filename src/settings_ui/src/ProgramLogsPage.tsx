import { memo, useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { Copy, Download, Pause, Play, Terminal } from "lucide-react";
import { Panel, Toggle } from "./components";
import { programLogHeaders } from "./programLogs";
import "./program-logs.css";

type LogSource = { id: string; label: string };
type LogChunk = { text: string; start: number; offset: number; version: string; reset: boolean; more: boolean; sources: LogSource[] };
type LogBuffer = { chunks: string[]; lines: string[]; columns: number; prependedLines: number };
const lineHeight = 20;

function emptyLogBuffer(): LogBuffer {
  return { chunks: [], lines: [""], columns: 0, prependedLines: 0 };
}

function addLogText(buffer: LogBuffer, text: string, prepend = false) {
  if (!text) return;
  let lines: string[];
  if (prepend) {
    buffer.chunks.unshift(text);
    lines = (text + (buffer.lines.shift() || "")).split("\n");
    buffer.lines = lines.concat(buffer.lines);
    buffer.prependedLines += lines.length - 1;
  } else {
    buffer.chunks.push(text);
    lines = ((buffer.lines.pop() || "") + text).split("\n");
    for (const line of lines) buffer.lines.push(line);
  }
  for (const line of lines) buffer.columns = Math.max(buffer.columns, line.replace(/\t/g, "    ").length);
}

const ProgramLogOutput = memo(function ProgramLogOutput({ buffer, revision, follow, onFollowChange, onEarlier }: {
  buffer: LogBuffer; revision: number; follow: boolean; onFollowChange: (value: boolean) => void; onEarlier: () => void;
}) {
  const output = useRef<HTMLDivElement>(null);
  const prependedLines = useRef(buffer.prependedLines);
  const [viewport, setViewport] = useState({ top: 0, height: 500 });
  useLayoutEffect(() => {
    const node = output.current;
    if (!node) return;
    const observer = new ResizeObserver(() => setViewport({ top: node.scrollTop, height: node.clientHeight }));
    observer.observe(node);
    return () => observer.disconnect();
  }, []);
  useLayoutEffect(() => {
    const node = output.current;
    if (!node) return;
    if (follow) node.scrollTop = node.scrollHeight;
    else node.scrollTop += Math.max(0, buffer.prependedLines - prependedLines.current) * lineHeight;
    prependedLines.current = buffer.prependedLines;
    setViewport(current => current.top === node.scrollTop && current.height === node.clientHeight
      ? current : { top: node.scrollTop, height: node.clientHeight });
  }, [revision, follow]);
  const first = Math.max(0, Math.min(buffer.lines.length - 1, Math.floor(viewport.top / lineHeight) - 20));
  const last = Math.min(buffer.lines.length, first + Math.ceil(viewport.height / lineHeight) + 40);
  return <div className="program-log-console" ref={output} tabIndex={0} aria-label="CMD 출력 원문" onScroll={() => {
    const node = output.current;
    if (!node) return;
    setViewport({ top: node.scrollTop, height: node.clientHeight });
    if (follow && node.scrollHeight - node.clientHeight - node.scrollTop > 40) onFollowChange(false);
    if (node.scrollTop < 200 && node.scrollHeight - node.clientHeight - node.scrollTop > 40) onEarlier();
  }} onWheel={event => { if (event.deltaY < 0 && output.current && output.current.scrollTop < 200) onEarlier(); }}>
    <div className="program-log-scroll-area" style={{ height: buffer.lines.length * lineHeight, minWidth: `${buffer.columns}ch` }}>
      <pre className="program-log-lines" style={{ top: first * lineHeight }}>{buffer.lines.slice(first, last).join("\n")}</pre>
    </div>
  </div>;
});

export function ProgramLogsPage({ notify }: { notify: (message: string) => void }) {
  const [source, setSource] = useState(() => new URLSearchParams(location.search).get("log") === "settings" ? "settings" : "illustration");
  const [sources, setSources] = useState<LogSource[]>([{ id: "illustration", label: "삽화·LoRA 실행" }, { id: "settings", label: "설정 서버" }]);
  const buffer = useRef<LogBuffer>(emptyLogBuffer());
  const [revision, setRevision] = useState(0);
  const [error, setError] = useState("");
  const [paused, setPaused] = useState(false);
  const [follow, setFollow] = useState(true);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [historyLoading, setHistoryLoading] = useState(false);
  const historyRequest = useRef<AbortController | null>(null);
  const cursor = useRef({ source: "", start: 0, offset: 0, version: "" });

  useEffect(() => {
    if (cursor.current.source !== source) {
      cursor.current = { source, start: 0, offset: 0, version: "" };
      buffer.current = emptyLogBuffer(); setRevision(value => value + 1);
      setError(""); setLoading(true);
    }
    if (paused) return;
    const controller = new AbortController();
    let timer: number;
    async function read() {
      let more = false;
      try {
        const query = new URLSearchParams({ source, offset: String(cursor.current.offset), version: cursor.current.version });
        const response = await fetch(`/api/setting/program-logs?${query}`, { headers: programLogHeaders, signal: controller.signal, cache: "no-store" });
        if (!response.ok) throw new Error(response.status === 404 ? "프로그램 로그를 연결하려면 프로그램을 다시 시작하세요." : `로그를 읽지 못했습니다. (${response.status})`);
        const data: LogChunk = await response.json();
        if (controller.signal.aborted) return;
        cursor.current = { source, start: data.reset ? data.start : cursor.current.start, offset: data.offset, version: data.version };
        setSources(data.sources); setError("");
        if (data.reset) {
          historyRequest.current?.abort(); historyRequest.current = null; setHistoryLoading(false);
          buffer.current = emptyLogBuffer();
          setLoading(true);
        }
        addLogText(buffer.current, data.text);
        if (data.reset || data.text) setRevision(value => value + 1);
        more = data.more;
        if (!more) setLoading(false);
      } catch (cause) {
        if (!controller.signal.aborted) { setError((cause as Error).message); setLoading(false); }
      } finally {
        if (!controller.signal.aborted) timer = window.setTimeout(read, more ? 10 : 1000);
      }
    }
    void read();
    return () => {
      controller.abort(); window.clearTimeout(timer);
      historyRequest.current?.abort(); historyRequest.current = null; setHistoryLoading(false);
    };
  }, [source, paused]);

  const readEarlier = useCallback(async () => {
    if (historyRequest.current || cursor.current.source !== source || !cursor.current.start) return;
    const controller = new AbortController();
    historyRequest.current = controller; setHistoryLoading(true);
    try {
      const query = new URLSearchParams({ source, before: String(cursor.current.start), version: cursor.current.version });
      const response = await fetch(`/api/setting/program-logs?${query}`, { headers: programLogHeaders, signal: controller.signal, cache: "no-store" });
      if (!response.ok) throw new Error(`이전 로그를 읽지 못했습니다. (${response.status})`);
      const data: LogChunk = await response.json();
      if (controller.signal.aborted || data.reset || data.version !== cursor.current.version) return;
      cursor.current.start = data.start;
      addLogText(buffer.current, data.text, true);
      setRevision(value => value + 1); setError("");
    } catch (cause) {
      if (!controller.signal.aborted) setError((cause as Error).message);
    } finally {
      if (historyRequest.current === controller) { historyRequest.current = null; setHistoryLoading(false); }
    }
  }, [source]);

  async function download() {
    setSaving(true);
    try {
      const response = await fetch(`/api/setting/program-logs/download?source=${encodeURIComponent(source)}`, { headers: programLogHeaders });
      if (!response.ok) throw new Error("로그를 저장하지 못했습니다.");
      const url = URL.createObjectURL(await response.blob());
      const anchor = document.createElement("a"); anchor.href = url; anchor.download = `${source}.log`;
      document.body.appendChild(anchor); anchor.click(); anchor.remove();
      window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (cause) { notify((cause as Error).message); }
    finally { setSaving(false); }
  }

  return <Panel title={<><Terminal size={16} />프로그램 출력</>} className="program-logs-panel">
    <div className="program-logs-toolbar">
      <label className="field">출력 창<select value={source} onChange={event => setSource(event.target.value)}>{sources.map(item => <option key={item.id} value={item.id}>{item.label}</option>)}</select></label>
      <button type="button" className="button secondary" onClick={() => setPaused(value => !value)}>{paused ? <Play size={14} /> : <Pause size={14} />}{paused ? "갱신 재개" : "갱신 일시정지"}</button>
      <div className="program-log-follow" data-enabled={follow}><span>자동 스크롤</span><Toggle checked={follow} onChange={setFollow} label="자동 스크롤" /></div>
      <div className="program-logs-actions"><button type="button" className="button secondary" disabled={!buffer.current.chunks.length} onClick={async () => {
        try { await navigator.clipboard.writeText(buffer.current.chunks.join("")); notify("로그 원문을 복사했습니다."); }
        catch { notify("복사하지 못했습니다. 로그를 선택해 복사하세요."); }
      }}><Copy size={14} />복사</button><button type="button" className="button secondary" disabled={saving} onClick={() => void download()}><Download size={14} />{saving ? "저장 중…" : "로그 저장"}</button></div>
    </div>
    <div className="program-logs-state" role="status">{error || (historyLoading ? "이전 로그 불러오는 중…" : paused ? "갱신 일시정지" : loading ? "로그 불러오는 중…" : "실시간 갱신 중")}</div>
    {loading || cursor.current.source !== source
      ? <div className="program-log-console" aria-label="CMD 출력 원문" aria-busy="true" />
      : <ProgramLogOutput key={source} buffer={buffer.current} revision={revision} follow={follow} onFollowChange={setFollow} onEarlier={readEarlier} />}
    {!buffer.current.chunks.length && !loading && !error && <p className="program-logs-state">아직 출력된 로그가 없습니다.</p>}
  </Panel>;
}
