import { useEffect, useRef, useState } from "react";
import { ArrowDownToLine, GitCommitHorizontal, RefreshCw, Square, Trash2 } from "lucide-react";
import { Badge, Modal, Panel } from "./components";
import "./updates.css";

type Component = { current: string | null; target: string | null; update_needed: boolean };
type UpdateStatus = {
  checked_at: string | null;
  application: Component & { ahead: number; behind: number; commits: string[]; restart_required: boolean };
  nodes: (Component & { name: string })[];
  comfy: Component & { version: string; required_version: string; required_ref: string; changes: string[] };
  backup: { bytes: number; path: string };
  job: { state: string; operation: string | null; phase_label: string; error: string | null; cancel_requested: boolean;
    logs: { seq: number; message: string; level: string }[]; result?: { message?: string; restart_required?: boolean } };
};
type Action = "fetch" | "application" | "soya" | "comfy" | "cancel" | "clear-backups";

async function requestUpdates(action?: Action): Promise<UpdateStatus> {
  const response = await fetch(`/api/setting/updates${action ? `/${action}` : ""}`, {
    method: action ? "POST" : "GET", headers: { "X-Comfy-Settings": "1" },
    cache: "no-store", signal: AbortSignal.timeout(30000),
  });
  const value = await response.json().catch(() => null);
  if (!response.ok) throw new Error(value?.error || (response.status === 404 ? "공방을 다시 실행해 업데이트 기능을 불러오세요." : "업데이트 요청을 처리하지 못했습니다."));
  if (!value?.job) throw new Error("업데이트 상태를 확인하지 못했습니다.");
  return value;
}

const short = (ref: string | null | undefined) => ref ? ref.slice(0, 12) : "미설치";
function bytes(size: number) {
  if (!size) return "0 B";
  const unit = Math.min(4, Math.floor(Math.log(size) / Math.log(1024)));
  return `${(size / 1024 ** unit).toLocaleString("ko-KR", { maximumFractionDigits: 1 })} ${["B", "KB", "MB", "GB", "TB"][unit]}`;
}

export function UpdatesPage() {
  const [status, setStatus] = useState<UpdateStatus | null>(null);
  const [pending, setPending] = useState<Action | null>(null);
  const [error, setError] = useState("");
  const [connectionError, setConnectionError] = useState("");
  const [confirmClear, setConfirmClear] = useState(false);
  const logOutput = useRef<HTMLDivElement>(null);
  const followOutput = useRef(true);
  useEffect(() => {
    if (pending) return;
    let disposed = false;
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      try {
        const value = await requestUpdates();
        if (!disposed) { setStatus(value); setConnectionError(""); }
      } catch (cause) {
        if (!disposed) setConnectionError(cause instanceof Error ? cause.message : "업데이트 상태를 가져오지 못했습니다.");
      } finally {
        if (!disposed) timer = setTimeout(poll, 2000);
      }
    }
    void poll();
    return () => { disposed = true; clearTimeout(timer); };
  }, [pending]);
  const job = status?.job;
  const running = job?.state === "running";
  const busy = !!pending || running;
  const available = !!status && !busy && !connectionError;
  const fetched = !!status?.checked_at;
  const application = status?.application;
  const ready = available && fetched && !application?.update_needed && !application?.restart_required;
  const ownJob = job?.operation?.startsWith("panel_");
  const lastLog = job?.logs.at(-1)?.seq;
  useEffect(() => {
    if (followOutput.current && logOutput.current) logOutput.current.scrollTop = logOutput.current.scrollHeight;
  }, [lastLog]);

  async function act(action: Action) {
    if (pending) return;
    setPending(action);
    setError("");
    try { setStatus(await requestUpdates(action)); setConnectionError(""); }
    catch (cause) { setError(cause instanceof Error ? cause.message : "업데이트 요청을 처리하지 못했습니다."); }
    finally { setPending(null); }
  }
  const actionLabel = (action: Action, label: string) => pending === action ? "요청 중…" : running && job?.operation === `panel_${action}` ? "진행 중…" : label;
  return <div className="stack updates-page">
    {(error || connectionError) && <p className="field-error" role="alert">{error || connectionError}</p>}
    <Panel title="1. 커밋 가져오기" aside={<GitCommitHorizontal size={17} />}>
      <div className="panel-body form-stack">
        <div className="updates-action-row">
          <p>공방과 Soya 커스텀 노드의 새 커밋을 확인합니다.</p>
          <button className="button primary" disabled={!available} onClick={() => void act("fetch")}><RefreshCw size={15} />{actionLabel("fetch", "커밋 가져오기")}</button>
        </div>
        <p className="field-help" role="status">{status?.checked_at ? `마지막 확인 · ${new Date(status.checked_at).toLocaleString("ko-KR")}` : status ? "커밋을 가져와 업데이트를 확인하세요." : "업데이트 상태 확인 중…"}</p>
        {!!application?.commits.length && <details className="updates-commits"><summary>새 커밋 {application.behind}개{application.behind > 20 ? " · 최근 20개 표시" : ""}</summary><ul>{application.commits.map(commit => <li key={commit}><code>{commit.split("\t")[0]}</code> {commit.split("\t").slice(1).join(" ")}</li>)}</ul></details>}
      </div>
    </Panel>
    <Panel title="2. 공방 업데이트">
      <div className="panel-body form-stack">
        <div className="updates-action-row">
          <div><p>공방 프로그램 · 내장 워크플로우 · 배포 프리셋</p><p className="field-help">현재 <code>{short(application?.current)}</code>{application?.target && <> · 가져온 커밋 <code>{short(application.target)}</code></>}</p></div>
          <button className="button primary" disabled={!available || !fetched || !application?.update_needed || application.restart_required || !!(application.ahead && application.behind)} onClick={() => void act("application")}><ArrowDownToLine size={15} />{actionLabel("application", "공방 업데이트")}</button>
        </div>
        <div className="updates-action-row"><span>내장 워크플로우 함께 업데이트</span><Badge tone="green">항상 활성화</Badge></div>
        {application?.restart_required ? <p role="status" className="field-help">공방을 종료한 뒤 다시 실행하고, 공방 업데이트를 눌러 내장 워크플로우 적용을 완료하세요.</p> : fetched && !application?.update_needed && <p role="status" className="field-help">{application?.ahead ? "현재 로컬 커밋이 가져온 커밋보다 앞서 있습니다." : "가져온 커밋이 적용되어 있습니다."}</p>}
        {!!(application?.ahead && application.behind) && <p className="field-error">로컬과 원격의 커밋 이력이 갈라져 있습니다. 저장소 이력을 확인하세요.</p>}
        <div className="updates-backup">
          <div className="updates-action-row"><div><strong>BACKUP</strong><p>현재 용량 <b>{status ? bytes(status.backup.bytes) : "확인 중…"}</b></p></div><button className="button secondary" disabled={!available || !status.backup.bytes} onClick={() => setConfirmClear(true)}><Trash2 size={15} />BACKUP 비우기</button></div>
          <p className="field-help">충돌한 파일의 수정본을 보관합니다. 프리셋 백업과 별도로 관리합니다.</p>
          {status && <p className="updates-path">{status.backup.path}</p>}
        </div>
      </div>
    </Panel>
    <Panel title="3. Soya 커스텀 노드 3종 업데이트">
      <div className="panel-body form-stack">
        <div className="updates-action-row"><p>Soya가 관리하는 커스텀 노드의 가져온 커밋을 적용합니다.</p><button className="button primary" disabled={!ready || !status?.nodes.some(node => node.update_needed)} onClick={() => void act("soya")}><ArrowDownToLine size={15} />{actionLabel("soya", "Soya 커스텀 노드 업데이트")}</button></div>
        <div className="updates-nodes">{status?.nodes.map(node => <div key={node.name} className="updates-node"><strong>{node.name}</strong><div><code>{short(node.current)}</code><span>→</span><code>{node.target ? short(node.target) : "커밋 가져오기 필요"}</code><Badge tone={node.update_needed ? "lavender" : "neutral"}>{!node.target ? "확인 전" : node.update_needed ? "업데이트 필요" : "적용됨"}</Badge></div></div>)}</div>
      </div>
    </Panel>
    <Panel title="4. Comfy 업데이트">
      <div className="panel-body form-stack">
        <div className="updates-action-row"><div><p>현재 공방 의존 버전 <strong>{status ? `ComfyUI ${status.comfy.required_version}` : "확인 중…"}</strong></p><p className="field-help">의존 커밋 <code>{status ? short(status.comfy.required_ref) : "…"}</code> · 설치 커밋 <code>{short(status?.comfy.current)}</code></p></div><button className="button primary" disabled={!ready || !status?.comfy.update_needed} onClick={() => void act("comfy")}><ArrowDownToLine size={15} />{actionLabel("comfy", "Comfy 업데이트")}</button></div>
        {fetched && <p className="field-help" role="status">가져온 공방의 의존 버전 · ComfyUI {status?.comfy.version} <code>{short(status?.comfy.target)}</code>{status?.comfy.update_needed ? " · 업데이트 필요" : " · 적용됨"}</p>}
        {fetched && !!status?.comfy.changes.length && <p className="field-help">업데이트 항목 · {status.comfy.changes.join(" · ")}</p>}
        {!status?.comfy.current && status && <p className="field-help">내장 ComfyUI를 먼저 설치하세요.</p>}
      </div>
    </Panel>
    {fetched && application?.update_needed && <p className="field-help">공방 업데이트 후 Soya 커스텀 노드와 Comfy를 업데이트하세요.</p>}
    {ownJob && job && <section className="install-console" aria-label="업데이트 콘솔">
      <div className="updates-action-row updates-console-heading"><strong>{running ? job.phase_label || "업데이트 진행 중…" : ({ succeeded: "완료", failed: "업데이트 실패", cancelled: "중단됨" }[job.state] || "업데이트")}</strong>{running && <button className="button secondary" disabled={!!pending || job.cancel_requested} onClick={() => void act("cancel")}><Square size={14} />{job.cancel_requested ? "중단 중…" : "작업 중단"}</button>}</div>
      <div className="install-console-output" role="log" aria-label="업데이트 로그" tabIndex={0} ref={logOutput} onScroll={event => { const el = event.currentTarget; followOutput.current = el.scrollHeight - el.scrollTop - el.clientHeight < 24; }}>{job.logs.map(entry => <div key={entry.seq} className={entry.level === "error" ? "install-console-error" : undefined}>{entry.message}</div>)}</div>
      {job.error && <p className="install-console-error" role="alert">{job.error}</p>}
    </section>}
    {confirmClear && <Modal title="BACKUP 비우기" onClose={() => setConfirmClear(false)}><div className="form-stack"><p>업데이트 BACKUP {status ? bytes(status.backup.bytes) : ""}를 영구 삭제합니다.</p><p>프리셋 백업은 유지됩니다.</p><div className="modal-actions"><button className="button secondary" onClick={() => setConfirmClear(false)}>취소</button><button className="button danger" disabled={!available} onClick={() => { setConfirmClear(false); void act("clear-backups"); }}>BACKUP 비우기</button></div></div></Modal>}
  </div>;
}
