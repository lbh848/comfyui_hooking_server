import { useEffect, useId, useRef, useState } from "react";
import { Download, Pause, Play, X } from "lucide-react";
import { Badge, Modal, Panel } from "./components";
import type { SaveState } from "./useSettings";
import { bytes } from "./formatBytes";
import { modelFolders } from "./data";

type DownloadFile = { id: string; name: string; remotePath: string; source: string; size: number; folder: string | null };
type DownloadGroup = { url: string; title?: string; source?: string; error?: string; requiredKind?: "models" | "loras"; versions: { id: string; name: string }[]; versionId?: string; files: DownloadFile[] };
type DownloadJob = { id: string; name: string; path: string; source?: string; state: string; downloaded: number; size: number; speed: number; error: string; warning?: string };
type DownloadStatus = { state: string; completed: number; total: number; items: DownloadJob[]; installBlockers: string[] };
const sources: Record<string, string> = { civitai: "Civitai", huggingface: "Hugging Face", civarchive: "CivArchive" };
const states: Record<string, string> = { queued: "대기", running: "다운로드 중", paused: "일시 정지", cancelled: "중단됨", completed: "다운로드 완료", failed: "실패" };
const folders = [["auto", "자동 · 모델 유형에 맞게"], ...modelFolders];
async function downloadRequest(action = "", body?: object) {
  const response = await fetch(`/api/setting/resources/download${action ? `/${action}` : ""}`, {
    method: body ? "POST" : "GET", cache: "no-store", signal: AbortSignal.timeout(120000),
    headers: { "X-Comfy-Settings": "1", ...(body ? { "Content-Type": "application/json" } : {}) },
    ...(body ? { body: JSON.stringify(body) } : {}),
  });
  const result = await response.json().catch(() => null);
  if (!response.ok || !result) throw new Error(result?.error || "다운로드 요청을 처리하지 못했습니다.");
  return result;
}

export function ResourceDownloads({ kind: initialKind, open, onClose, onInstalled, isSaved, saveState, hasCivitaiKey }: {
  kind: "models" | "loras"; open: boolean; onClose: () => void; onInstalled: () => Promise<void>;
  isSaved: boolean; saveState: SaveState; hasCivitaiKey: boolean;
}) {
  const [kind, setKind] = useState(initialKind);
  const [urls, setUrls] = useState("");
  const [groups, setGroups] = useState<DownloadGroup[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [folder, setFolder] = useState("auto");
  const [subfolder, setSubfolder] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [statusError, setStatusError] = useState("");
  const [statusReady, setStatusReady] = useState(false);
  const [status, setStatus] = useState<DownloadStatus>({ state: "idle", completed: 0, total: 0, items: [], installBlockers: [] });
  const resolveBlockersId = useId();
  const installBlockersId = useId();
  const queueBusyId = useId();
  const completed = useRef("");
  const completedFiles = status.items.filter(item => item.state === "completed").map(item => item.id).join(",");
  const label = kind === "loras" ? "LoRA" : "모델";
  const active = status.state === "running";
  useEffect(() => {
    if (busy) return;
    let disposed = false;
    let timer: ReturnType<typeof setTimeout>;
    async function readStatus() {
      let retry = false;
      try {
        const next = await downloadRequest();
        if (!Array.isArray(next.installBlockers)) {
          retry = true;
          if (!disposed) setStatusError("설정 서버를 재시작하고 화면을 새로고침하세요.");
          return;
        }
        if (!disposed) { setStatus(next); setStatusReady(true); setStatusError(""); }
      } catch {
        retry = true;
        if (!disposed) setStatusError("다운로드 상태를 불러오지 못했습니다. 서버 연결을 확인하세요.");
      } finally {
        if (!disposed && (open || active || retry)) timer = setTimeout(readStatus, 1000);
      }
    }
    void readStatus();
    return () => { disposed = true; clearTimeout(timer); };
  }, [open, active, busy]);
  useEffect(() => {
    if (completedFiles && completedFiles !== completed.current) void onInstalled();
    completed.current = completedFiles;
  }, [completedFiles, onInstalled]);

  function defaultSelection(next: DownloadGroup[]) {
    return next.flatMap(group => group.source === "huggingface" ? group.files.map(file => file.id) : group.files.slice(0, 1).map(file => file.id));
  }
  async function resolve(nextKind = kind) {
    setBusy(true); setError("");
    if (nextKind !== kind) {
      setKind(nextKind); setGroups([]); setSelected(new Set()); setFolder("auto");
    }
    try {
      const result = await downloadRequest("resolve", { urls: urls.split("\n").map(value => value.trim()).filter(Boolean), kind: nextKind });
      setGroups(result.groups);
      setSelected(new Set(defaultSelection(result.groups)));
    } catch (cause) { setError(cause instanceof Error ? cause.message : "주소를 확인하세요."); }
    finally { setBusy(false); }
  }
  async function changeVersion(index: number, versionId: string) {
    setBusy(true); setError("");
    try {
      const previous = groups[index];
      const result = await downloadRequest("resolve", { urls: [previous.url], kind, versionId });
      const group: DownloadGroup = { ...result.groups[0], versions: previous.versions };
      setGroups(current => current.map((value, position) => position === index ? group : value));
      setSelected(current => new Set([...current].filter(id => !previous.files.some(file => file.id === id)).concat(defaultSelection([group]))));
    } catch (cause) { setError(cause instanceof Error ? cause.message : "버전을 확인하세요."); }
    finally { setBusy(false); }
  }
  async function action(name: string, body: object = {}) {
    setBusy(true); setError("");
    try { setStatus(await downloadRequest(name, body)); }
    catch (cause) { setError(cause instanceof Error ? cause.message : "다운로드 요청에 실패했습니다."); }
    finally { setBusy(false); }
  }
  const availableFiles = groups.flatMap(group => group.files);
  const files = availableFiles.filter(file => selected.has(file.id));
  const civitaiRequested = files.some(file => file.source === "civitai") || urls.split("\n").some(value => {
    value = value.trim();
    if (/^\d+$/.test(value)) return true;
    try { return ["civitai.com", "www.civitai.com", "civitai.red", "civitai.green"].includes(new URL(value).hostname); }
    catch { return false; }
  });
  const missingCivitaiKey = !hasCivitaiKey && "Civitai API 키 미입력: 기타 API 관리에서 키를 입력하세요.";
  const civitaiBlocker = civitaiRequested && missingCivitaiKey;
  const settingsBlocker = !isSaved && (saveState === "error" || saveState === "conflict"
    ? "설정이 저장되지 않았습니다. 창을 닫고 상단 오류 안내를 확인하세요." : "설정 저장 중…");
  const resolveBlockers = [busy && "요청 처리 중…", settingsBlocker, civitaiBlocker, !urls.trim() && "모델 주소를 입력하세요."].filter(Boolean);
  const selectionBlocker = !urls.trim() ? "모델 주소를 입력하세요."
    : !groups.length ? "파일 정보 불러오기를 먼저 실행하세요."
    : !availableFiles.length ? "설치할 수 있는 파일이 없습니다. 주소별 오류를 확인하세요."
    : !files.length ? "설치할 파일을 선택하세요." : "";
  const installBlockers = [
    busy && "요청 처리 중…",
    active && "다운로드가 진행 중입니다. 완료 후 다시 시도하세요.",
    statusError || (!statusReady && "다운로드 상태 확인 중…"),
    ...status.installBlockers,
    settingsBlocker,
    civitaiBlocker,
    selectionBlocker,
  ].filter(Boolean);
  const resumeBlockers = [
    busy && "요청 처리 중…", ...status.installBlockers, settingsBlocker,
    status.items.some(item => item.state !== "completed" && item.source === "civitai") && missingCivitaiKey,
  ].filter(Boolean);
  const resumable = ["paused", "failed", "cancelled"].includes(status.state);
  const queue = status.total > 0 && <Panel title="다운로드" aside={<div className="download-actions">
    <Badge>{states[status.state] || status.state}</Badge>
    {status.state === "completed" && <button type="button" className="icon-button" disabled={busy} aria-label="완료된 다운로드 목록 비우기" title="완료된 다운로드 목록 비우기" onClick={() => void action("clear")}><X size={16} /></button>}
  </div>}>
    <div className="panel-body form-stack">
      <div className="download-queue-summary"><span>{status.completed} / {status.total}개 다운로드 완료</span>
        <div className="download-actions">
          {active && <><button className="button secondary small" disabled={busy} aria-describedby={busy ? queueBusyId : undefined} onClick={() => void action("pause")}><Pause size={13} />일시 정지</button><button className="button secondary small" disabled={busy} aria-describedby={busy ? queueBusyId : undefined} onClick={() => void action("cancel")}><X size={13} />중단</button></>}
          {resumable && <button className="button secondary small" disabled={resumeBlockers.length > 0} aria-describedby={queueBusyId} onClick={() => void action("resume")}><Play size={13} />{status.state === "paused" ? "이어받기" : "다시 시도"}</button>}
        </div>
      </div>
      {(busy || resumable) && <p id={queueBusyId} className="field-error" role="status">{resumable ? resumeBlockers.join(" · ") : "요청 처리 중…"}</p>}
      <div className="download-queue-list">{status.items.map(item => <div className="download-job" key={item.id}>
        <div><strong>{item.name}</strong><Badge>{states[item.state] || item.state}</Badge></div>
        <small>{item.path}</small>
        <progress aria-label={`${item.name} 다운로드 진행률`} value={item.downloaded} max={item.size || 1} />
        <small>{bytes(item.downloaded)} / {bytes(item.size)}{item.speed > 0 && ` · ${bytes(item.speed)}/s`}</small>
        {item.error && <p className="field-error" role="alert">{item.error}</p>}
        {item.warning && <p className="field-error" role="alert">{item.warning}</p>}
      </div>)}</div>
    </div>
  </Panel>;
  return <>
    {!open && status.installBlockers.length > 0 && <p className="field-error" role="status">{status.installBlockers.join(" · ")}</p>}
    {!open && queue}
    {!open && statusError && <p className="field-error" role="alert">{statusError}</p>}
    {!open && error && <p className="field-error" role="alert">{error}</p>}
    {open && <Modal title={`${label} 설치`} onClose={onClose} working={busy || active}>
      <div className="form-stack resource-download-form">
        <p className="dialog-description">Civitai, Hugging Face, CivArchive 주소를 한 줄에 하나씩 입력하세요.</p>
        <label className="field">모델 주소<textarea rows={3} value={urls} onChange={event => { setUrls(event.target.value); setGroups([]); setSelected(new Set()); }} placeholder="https://civitai.com/models/…&#10;https://huggingface.co/사용자/저장소" /></label>
        <div className="download-actions download-resolve-actions">
          <button className="button secondary" disabled={resolveBlockers.length > 0} aria-describedby={resolveBlockersId} onClick={() => void resolve()}>{busy ? "처리 중…" : "파일 정보 불러오기"}</button>
          <p id={resolveBlockersId} className="field-error" role="status" aria-atomic="true">{resolveBlockers.join(" · ")}</p>
        </div>
        {groups.length > 0 && <>
          <div className="download-file-groups">{groups.map((group, index) => <section className="download-file-group" key={`${index}-${group.url}`}>
            <header><strong>{group.title || group.url}</strong>{group.source && <Badge>{sources[group.source]}</Badge>}</header>
            {group.error && <p className="field-error" role="alert">{group.error}</p>}
            {group.requiredKind && group.requiredKind !== kind && <button type="button" className="button secondary small" disabled={resolveBlockers.length > 0 || active} onClick={() => void resolve(group.requiredKind)}>{group.requiredKind === "loras" ? "LoRA" : "모델"} 설치로 전환</button>}
            {group.versions.length > 1 && <label className="field">버전<select value={group.versionId} disabled={busy} onChange={event => void changeVersion(index, event.target.value)}>{group.versions.map(version => <option key={version.id} value={version.id}>{version.name}</option>)}</select></label>}
            {group.files.map(file => <label className="download-file" key={file.id}>
              <input type="checkbox" checked={selected.has(file.id)} disabled={busy} onChange={event => setSelected(current => {
                const next = new Set(current);
                if (event.target.checked) next.add(file.id); else next.delete(file.id);
                return next;
              })} />
              <span><strong>{file.remotePath}</strong><small>{bytes(file.size)}{file.folder && ` · ${file.folder}`}</small></span>
            </label>)}
          </section>)}</div>
          <div className="download-selection-actions"><button className="text-button" onClick={() => setSelected(new Set(groups.flatMap(group => group.files.map(file => file.id))))}>전체 선택</button><button className="text-button" onClick={() => setSelected(new Set())}>선택 해제</button><span>{files.length}개 · {bytes(files.reduce((sum, file) => sum + file.size, 0))}</span></div>
          <div className="two-columns">
            <label className="field">설치 폴더<select value={kind === "loras" ? "loras" : folder} onChange={event => setFolder(event.target.value)}>{kind === "loras" ? <option value="loras">LoRA · loras</option> : folders.map(([value, name]) => <option value={value} key={value}>{name}</option>)}</select></label>
            <label className="field">하위 폴더 (선택)<input value={subfolder} onChange={event => setSubfolder(event.target.value)} placeholder="예: illustration" /></label>
          </div>
          <p className="field-help">저장 위치: data/comfy/models/{kind === "loras" ? "loras" : folder === "auto" ? "모델 유형별 폴더" : folder}/{subfolder && `${subfolder}/`}</p>
        </>}
        {queue}
        {error && <p className="field-error" role="alert">{error}</p>}
        <div className="modal-actions download-install-actions">
          <p id={installBlockersId} className="field-error" role="status" aria-atomic="true">{installBlockers.join(" · ")}</p>
          <button className="button secondary" onClick={onClose}>닫기</button>
          <button className="button primary" disabled={installBlockers.length > 0} aria-describedby={installBlockersId} onClick={() => void action("", { files: [...selected], kind, folder, subfolder })}><Download size={15} />선택한 파일 설치</button>
        </div>
      </div>
    </Modal>}
  </>;
}
