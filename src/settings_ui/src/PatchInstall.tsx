import { useId, useState } from "react";
import { PackagePlus } from "lucide-react";
import { Panel } from "./components";

type PatchStart = { upload_id: string; chunk_size: number };
type PatchChunk = { received: number };
type PatchResult = {
  patch_name: string;
  summary: {
    characters: { added: number; skipped_equal: number; skipped_existing: number };
    assets: { added_files: number; skipped_existing_files: number };
  };
};

async function requestPatch<T>(path: string, data: Record<string, string | number> | Blob): Promise<T> {
  const chunk = data instanceof Blob;
  const response = await fetch(`/api/patch-import/upload/${path}`, {
    method: "POST",
    headers: {
      "X-Comfy-Settings": "1",
      "X-Comfy-Settings-Schema": "2",
      "Content-Type": chunk ? "application/octet-stream" : "application/json",
    },
    body: chunk ? data : JSON.stringify(data),
    signal: path === "complete" ? undefined : AbortSignal.timeout(30000),
  }).catch(() => { throw new Error("설정 서버에 연결할 수 없습니다. 연결을 확인하고 다시 시도하세요."); });
  const value = await response.json().catch(() => null);
  if (!response.ok || !value?.ok) throw new Error(value?.error || "패치 설치 요청을 처리하지 못했습니다.");
  return value;
}

export function PatchInstall() {
  const [file, setFile] = useState<File | null>(null);
  const [progress, setProgress] = useState<number | null>(null);
  const [result, setResult] = useState<PatchResult | null>(null);
  const [error, setError] = useState("");
  const helpId = useId();
  const busy = progress !== null;

  async function installPatch() {
    if (!file || busy) return;
    if (!file.name.toLowerCase().endsWith(".soyapatch")) {
      setError(".soyapatch 파일을 선택하세요.");
      return;
    }
    setProgress(0);
    setResult(null);
    setError("");
    let uploadId = "";
    try {
      const started = await requestPatch<PatchStart>("start", { filename: file.name, file_size: file.size });
      uploadId = started.upload_id;
      for (let offset = 0; offset < file.size;) {
        const end = Math.min(file.size, offset + started.chunk_size);
        const query = new URLSearchParams({ upload_id: uploadId, offset: String(offset) });
        const { received } = await requestPatch<PatchChunk>(`chunk?${query}`, file.slice(offset, end));
        if (received !== end || received <= offset) throw new Error("패치 파일의 전송 크기가 일치하지 않습니다. 다시 시도하세요.");
        setProgress(received);
        offset = received;
      }
      setResult(await requestPatch<PatchResult>("complete", { upload_id: uploadId }));
    } catch (cause) {
      if (uploadId) void requestPatch("abort", { upload_id: uploadId }).catch(() => {});
      setError(cause instanceof Error ? cause.message : "패치를 설치하지 못했습니다.");
    } finally {
      setProgress(null);
    }
  }

  return (
    <Panel title="패치 설치">
      <form className="panel-body form-stack patch-install-form" onSubmit={event => { event.preventDefault(); void installPatch(); }}>
        <label className="field">
          <span>패치 파일</span>
          <input type="file" accept=".soyapatch" disabled={busy} aria-describedby={helpId} onChange={event => {
            setFile(event.target.files?.[0] ?? null);
            setResult(null);
            setError("");
          }} />
        </label>
        <p id={helpId} className="field-help">다운로드한 .soyapatch 파일을 선택하세요.</p>
        <div className="install-actions">
          <button type="submit" className="button secondary" disabled={!file || busy}>
            <PackagePlus size={16} aria-hidden="true" />패치 설치
          </button>
        </div>
        {progress !== null && file && <>
          <progress className="install-progress" aria-label="패치 업로드 진행" max={file.size} value={progress} />
          <p className="field-help" role="status">{progress < file.size ? "패치 파일 업로드 중…" : "패치를 확인하고 설치하는 중…"}</p>
        </>}
        {error && <p className="field-error" role="alert">{error}</p>}
        {result && <div className="field-help" role="status">
          <p>{result.patch_name || file?.name} 패치를 설치했습니다.</p>
          {(result.summary.characters.added + result.summary.characters.skipped_equal + result.summary.characters.skipped_existing > 0) &&
            <p>캐릭터 {result.summary.characters.added}개 추가 · 기존 캐릭터 {result.summary.characters.skipped_equal + result.summary.characters.skipped_existing}개 유지</p>}
          {(result.summary.assets.added_files + result.summary.assets.skipped_existing_files > 0) &&
            <p>갤러리 파일 {result.summary.assets.added_files}개 추가 · 기존 파일 {result.summary.assets.skipped_existing_files}개 유지</p>}
        </div>}
      </form>
    </Panel>
  );
}
