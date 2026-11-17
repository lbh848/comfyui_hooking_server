import { useEffect, useState, type FormEvent } from "react";
import { FolderSearch, Link2, Unlink } from "lucide-react";
import { Badge, Empty, Modal } from "./components";
import { bytes } from "./formatBytes";

type LinkedFolder = { id: string; path: string; source: string; folder: string; available: boolean };
type SourceFolder = { folder: string; label: string; source: string; count: number; size: string; connected: boolean };
type Inspection = { source: string; folders: SourceFolder[] };

async function linkRequest(body?: Record<string, unknown>, linkId?: string, signal?: AbortSignal) {
  const response = await fetch(`/api/setting/resources/links${linkId ? `/${encodeURIComponent(linkId)}` : ""}`, {
    method: linkId ? "DELETE" : body ? "POST" : "GET", cache: "no-store",
    headers: { "X-Comfy-Settings": "1", "Content-Type": "application/json" },
    ...(body && { body: JSON.stringify(body) }), signal,
  });
  const result = await response.json().catch(() => null);
  if (!response.ok || !result) throw new Error(result?.error || "모델 연결 정보를 읽지 못했습니다.");
  return result;
}

export function ResourceLinkedUpload({ kind, onClose, onLinked, notify }: {
  kind: "models" | "loras"; onClose: () => void; onLinked: () => Promise<void>; notify: (message: string) => void;
}) {
  const label = kind === "loras" ? "LoRA" : "모델";
  const [source, setSource] = useState("");
  const [inspection, setInspection] = useState<Inspection | null>(null);
  const [selected, setSelected] = useState<string[]>([]);
  const [links, setLinks] = useState<LinkedFolder[]>([]);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [disconnecting, setDisconnecting] = useState("");
  useEffect(() => {
    const controller = new AbortController();
    void linkRequest(undefined, undefined, controller.signal).then(result => setLinks(result.links))
      .catch(cause => { if (!controller.signal.aborted) setError(cause.message); })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, []);

  async function inspect(event: FormEvent) {
    event.preventDefault();
    if (busy) return;
    setBusy(true); setError(""); setInspection(null); setSelected([]);
    try {
      const result: Inspection = await linkRequest({ action: "scan", source: source.trim(), kind });
      setInspection(result);
      setSelected(result.folders.filter(folder => !folder.connected).map(folder => folder.folder));
    } catch (cause) { setError(cause instanceof Error ? cause.message : "외부 모델 폴더를 읽지 못했습니다."); }
    finally { setBusy(false); }
  }
  async function connect() {
    if (!inspection || !selected.length || busy) return;
    setBusy(true); setError("");
    try {
      const result = await linkRequest({ action: "connect", source: inspection.source, kind, folders: selected });
      notify(`${label} 폴더 ${result.connected}개를 연결했습니다.`);
      await onLinked(); onClose();
    } catch (cause) { setError(cause instanceof Error ? cause.message : "모델 폴더를 연결하지 못했습니다."); }
    finally { setBusy(false); }
  }
  async function disconnect(linkId: string) {
    if (busy) return;
    setBusy(true); setError("");
    try {
      await linkRequest(undefined, linkId);
      setLinks(current => current.filter(link => link.id !== linkId));
      setDisconnecting(""); setInspection(null); setSelected([]);
      notify("연결을 해제했습니다. 외부 원본 파일은 유지됩니다.");
      await onLinked();
    } catch (cause) { setError(cause instanceof Error ? cause.message : "연결을 해제하지 못했습니다."); }
    finally { setBusy(false); }
  }
  const visibleLinks = links.filter(link => (link.folder === "loras") === (kind === "loras"));
  const selectedFolders = inspection?.folders.filter(folder => selected.includes(folder.folder)) || [];
  return <Modal title={`${label} 링크드 업로드`} onClose={onClose} closeDisabled={busy || loading}>
    <div className="form-stack resource-linked-upload" aria-busy={busy || loading}>
      <p className="dialog-description">외부 ComfyUI의 {label} 폴더를 연결합니다. 파일을 복사하지 않고 기존 파일과 함께 사용하며, 연결된 파일에는 LINKED가 표시됩니다.</p>
      <form className="form-stack" onSubmit={event => void inspect(event)}>
        <label className="field">외부 ComfyUI 경로<input autoFocus required value={source} disabled={busy || loading} placeholder="예: D:\ComfyUI 또는 D:\ComfyUI\models" onChange={event => { setSource(event.target.value); setInspection(null); setSelected([]); setError(""); }} /><span className="field-help">이 컴퓨터에 설치된 ComfyUI 폴더 또는 models 폴더의 전체 경로를 입력하세요.</span></label>
        <div className="download-actions"><button className="button secondary" type="submit" disabled={busy || loading || !source.trim()}><FolderSearch size={15} />{busy ? "처리 중…" : "폴더 확인"}</button></div>
      </form>
      {inspection && <section className="form-stack" aria-label="연결할 폴더">
        <p className="resource-file-path">{inspection.source}</p>
        {inspection.folders.length ? <>
          <div className="download-selection-actions"><button className="button secondary small" disabled={busy} onClick={() => setSelected(inspection.folders.filter(folder => !folder.connected).map(folder => folder.folder))}>전체 선택</button><button className="button secondary small" disabled={busy} onClick={() => setSelected([])}>선택 해제</button></div>
          <div className="resource-linked-folders">{inspection.folders.map(folder => <label className="resource-linked-folder" key={folder.folder}>
            <input type="checkbox" checked={selected.includes(folder.folder)} disabled={busy || folder.connected} onChange={event => setSelected(current => event.target.checked ? [...current, folder.folder] : current.filter(value => value !== folder.folder))} />
            <span><strong>{folder.label} · {folder.folder}</strong><small>{folder.count}개 · {bytes(Number(folder.size))}{folder.connected && " · 이미 연결됨"}</small></span>
          </label>)}</div>
          <p className="field-help">선택한 폴더 {selected.length}개 · 파일 {selectedFolders.reduce((total, folder) => total + folder.count, 0)}개 · {bytes(selectedFolders.reduce((total, folder) => total + Number(folder.size), 0))}<br />폴더 안에 추가하거나 제거한 파일도 목록을 새로고침하면 반영됩니다.</p>
        </> : <Empty title={`연결할 ${label} 파일이 없습니다`}>선택한 models 폴더를 확인하세요.</Empty>}
      </section>}
      <p className="field-help">외부 원본 폴더를 이동하거나 드라이브 연결을 해제하면 사용할 수 없습니다. 연결 해제는 폴더 단위이며 외부 원본 파일은 삭제하지 않습니다.</p>
      <section className="form-stack" aria-label="연결된 폴더">
        <h3>연결된 폴더</h3>
        {loading && <p role="status">연결 정보를 읽는 중…</p>}
        {!loading && !visibleLinks.length && <p className="field-help">연결된 폴더가 없습니다.</p>}
        {visibleLinks.map(link => <div className="resource-linked-connection" key={link.id}>
          <div className="download-actions"><Badge tone="lavender">LINKED</Badge><strong>{link.folder}</strong><span className="field-help">{link.available ? "연결됨" : "원본 폴더를 찾을 수 없음"}</span></div>
          <p className="resource-file-path">{link.source}</p>
          {disconnecting === link.id ? <><p className="field-help">이 폴더의 모든 {label} 연결을 해제하시겠습니까? 외부 원본 파일은 유지됩니다.</p><div className="download-actions"><button className="button secondary small" disabled={busy} onClick={() => setDisconnecting("")}>취소</button><button className="button danger small" disabled={busy} onClick={() => void disconnect(link.id)}>해제 확인</button></div></>
            : <button className="button secondary small" disabled={busy} onClick={() => setDisconnecting(link.id)}><Unlink size={13} />연결 해제</button>}
        </div>)}
      </section>
      {error && <p className="field-error" role="alert">{error}</p>}
      <div className="modal-actions"><button className="button secondary" disabled={busy || loading} onClick={onClose}>닫기</button><button className="button primary" disabled={busy || loading || !selected.length} onClick={() => void connect()}><Link2 size={15} />{busy ? "처리 중…" : "선택한 폴더 연결"}</button></div>
    </div>
  </Modal>;
}
