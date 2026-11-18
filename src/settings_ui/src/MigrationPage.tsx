import { useEffect, useState } from "react";
import { FolderSearch, Import } from "lucide-react";
import { Panel } from "./components";
import { bytes } from "./formatBytes";
import "./migration.css";

const categories = {
  asset_lora: "에셋 LoRA",
  bot_lora: "봇 LoRA",
  style_lora: "그림체 LoRA",
  bot: "봇",
  asset: "에셋",
} as const;
type Kind = keyof typeof categories;
type Item = { id: string; kind: Kind; label: string; target: string; detail: string; imported: boolean; size: number };
type Result = { id: string; label: string; target: string; state: string; message: string };
type Job = { id?: string; state: "idle" | "running" | "finished" | "stopped"; source?: string; total: number; bytes: number; current: string; results: Result[] };
type Status = { default_source?: string; job: Job };

async function request<T>(action = "", body?: unknown, signal?: AbortSignal): Promise<T> {
  const response = await fetch("/api/setting/migration" + (action && "/" + action), {
    method: body === undefined ? "GET" : "POST", cache: "no-store", signal,
    headers: { "X-Comfy-Settings": "1", "X-Comfy-Settings-Schema": "2", "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const value = await response.json().catch(() => null);
  if (!response.ok || !value) throw new Error(value?.error || (response.status === 404
    ? "공방을 다시 실행해 이사하기를 불러오세요." : "이사 요청을 처리하지 못했습니다."));
  return value;
}

export function MigrationPage() {
  const [source, setSource] = useState<string | null>(null);
  const [previousJob, setPreviousJob] = useState<string>();
  const [scan, setScan] = useState<{ source: string; items: Item[] } | null>(null);
  const [chosen, setChosen] = useState<string[]>([]);
  const [job, setJob] = useState<Job | null>(null);
  const [pending, setPending] = useState("");
  const [error, setError] = useState("");
  const [connectionError, setConnectionError] = useState("");
  const running = job?.state === "running";
  const busy = !!pending || running;

  useEffect(() => {
    const controller = new AbortController();
    let timer: number;
    async function poll() {
      try {
        const value = await request<Status>("", undefined, controller.signal);
        if (controller.signal.aborted) return;
        setJob(value.job);
        setSource(current => current ?? value.job.source ?? value.default_source ?? "");
        setConnectionError("");
      } catch (e) {
        if (!controller.signal.aborted) setConnectionError((e as Error).message);
      } finally {
        if (!controller.signal.aborted) timer = window.setTimeout(poll, running ? 800 : 3500);
      }
    }
    if (!pending) void poll();
    return () => { controller.abort(); window.clearTimeout(timer); };
  }, [pending, running]);

  const completed = new Set(job?.id !== previousJob ? job?.results.filter(r => r.state !== "error").map(r => r.id) : []);
  const available = (item: Item) => !item.imported && !completed.has(item.id);
  const selected = scan?.items.filter(i => available(i) && chosen.includes(i.id)) || [];
  function choose(items: Item[], checked: boolean) {
    const ids = new Set(items.filter(available).map(i => i.id));
    setChosen(previous => checked ? [...new Set([...previous, ...ids])] : previous.filter(id => !ids.has(id)));
  }
  async function act(action: "scan" | "start" | "stop") {
    setPending(action); setError("");
    try {
      if (action === "scan") {
        setScan(null); setChosen([]);
        const result = await request<{ source: string; items: Item[] }>("scan", { source });
        setPreviousJob(job?.id);
        setScan(result);
      } else {
        const body = action === "stop" ? {} : { source: scan?.source, selection: Object.fromEntries(selected.map(i => [i.id, i.target])) };
        setJob((await request<Status>(action, body)).job);
      }
    } catch (e) { setError((e as Error).message); }
    finally { setPending(""); }
  }
  return <div className="stack migration-page">
    <Panel title="원본 프로젝트">
      <form className="panel-body migration-source" onSubmit={e => { e.preventDefault(); if (!busy && source?.trim()) void act("scan"); }}>
        <label className="field">원본 프로젝트 폴더<input value={source ?? ""} disabled={busy}
          onChange={e => { setSource(e.target.value); setScan(null); setChosen([]); }}
          placeholder="asset_data가 들어 있는 폴더의 전체 경로" /></label>
        <button className="button" type="submit" disabled={busy || !source?.trim()}><FolderSearch size={16} />{pending === "scan" ? "조회 중…" : "목록 조회"}</button>
      </form>
    </Panel>
    {(error || connectionError) && <p className="field-error" role="alert">{error || connectionError}</p>}
    {scan && <Panel title="가져올 항목" aside={<span>{selected.length}개 선택 · {bytes(selected.reduce((total, item) => total + item.size, 0))}</span>}>
      <div className="panel-body form-stack">
        {(Object.keys(categories) as Kind[]).map(kind => {
          const rows = scan.items.filter(item => item.kind === kind);
          const ready = rows.filter(available);
          const count = selected.filter(i => i.kind === kind).length;
          return <details key={kind} className="migration-group">
            <summary>{categories[kind]} <small>{rows.length}개{count > 0 && ` · ${count}개 선택`}</small></summary>
            {rows.length ? <>
              {kind === "asset" && <p>캐릭터의 이미지와 그룹을 가져옵니다. 프리셋은 제외합니다.</p>}
              {kind === "bot" && <p>캐릭터 카드, 이미지, 봇 지시문을 가져옵니다. 카드의 LoRA는 현재 공방에서 파일을 찾을 수 있을 때 연결합니다.</p>}
              {kind === "bot_lora" && <p>봇 단위로 가져옵니다. 가져온 프로젝트는 ‘연결 끊김’으로 표시됩니다.</p>}
              <label className="migration-check"><input type="checkbox" disabled={busy || !ready.length}
                checked={!!ready.length && count === ready.length} ref={el => { if (el) el.indeterminate = count > 0 && count < ready.length; }}
                onChange={e => choose(rows, e.target.checked)} />{categories[kind]} 전체 선택</label>
              <div className="migration-items">{rows.map(item => <label className="migration-check" key={item.id}>
                <input type="checkbox" checked={available(item) && chosen.includes(item.id)} disabled={busy || !available(item)}
                  onChange={e => choose([item], e.target.checked)} />
                <span><strong>{item.label}</strong>{item.label !== item.target && <small>가져올 이름 · {item.target}</small>}
                  <small>{[item.detail, available(item) ? bytes(item.size) : "이미 가져옴"].filter(Boolean).join(" · ")}</small></span>
              </label>)}</div></> : <p>가져올 항목이 없습니다.</p>}
          </details>;
        })}
        <div><button className="button primary" disabled={busy || !selected.length || !!connectionError}
          onClick={() => void act("start")}><Import size={16} />선택 항목 가져오기{selected.length > 0 && ` (${selected.length})`}</button></div>
      </div>
    </Panel>}
    {job && job.state !== "idle" && <Panel title="가져오기 결과" aside={running &&
      <button className="button" disabled={!!pending} onClick={() => void act("stop")}>중단</button>}>
      <div className="panel-body form-stack">
        <strong role="status">{running ? "가져오는 중" : job.state === "stopped" ? "중단됨" : "처리 완료"} · {job.results.length} / {job.total}개 · {bytes(job.bytes)}</strong>
        {job.current && <span>{job.current}</span>}
        {job.results.map(r => <div key={r.id} className={r.state === "error" ? "field-error" : ""}>
          <strong>{r.label}</strong> · {r.state === "error" && "실패 · "}{r.message}</div>)}
      </div>
    </Panel>}
  </div>;
}
