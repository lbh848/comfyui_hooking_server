import { useEffect, useRef, useState, type FormEvent } from "react";
import { Upload } from "lucide-react";
import { Modal } from "./components";
import { modelFolders } from "./data";
import { bytes } from "./formatBytes";

const modelExtensions = ".safetensors,.sft,.ckpt,.pt,.pth,.bin,.gguf,.onnx,.pkl,.torchscript";

export function ResourceUpload({ kind, onClose, onUploaded, notify }: {
  kind: "models" | "loras"; onClose: () => void; onUploaded: () => Promise<void>; notify: (message: string) => void;
}) {
  const label = kind === "loras" ? "LoRA" : "모델";
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [folder, setFolder] = useState(kind === "loras" ? "loras" : "checkpoints");
  const [subfolder, setSubfolder] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [progress, setProgress] = useState({ loaded: 0, total: 0 });
  const request = useRef<XMLHttpRequest | null>(null);
  const previewInput = useRef<HTMLInputElement>(null);
  useEffect(() => () => request.current?.abort(), []);
  useEffect(() => {
    if (!preview) { setPreviewUrl(""); return; }
    const url = URL.createObjectURL(preview);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [preview]);

  async function upload(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (busy) return;
    setError("");
    if (!file?.size) { setError("업로드할 파일을 선택하세요. 빈 파일은 업로드할 수 없습니다."); return; }
    if (preview && (!preview.size || preview.size > 10 * 1024 * 1024)) {
      setError("미리보기 이미지는 빈 파일이 아닌 10 MB 이하 파일로 선택하세요."); return;
    }
    const values = new FormData(event.currentTarget);
    const body = new FormData();
    body.append("kind", kind);
    body.append("folder", folder);
    body.append("subfolder", subfolder.trim());
    body.append("family", String(values.get("family") ?? "").trim());
    body.append("trigger", String(values.get("trigger") ?? "").trim());
    body.append("size", String(file.size));
    body.append("file", file, file.name);
    if (preview) body.append("preview", preview, preview.name);
    setBusy(true);
    setProgress({ loaded: 0, total: 0 });
    try {
      await new Promise<void>((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        request.current = xhr;
        xhr.open("POST", "/api/setting/resources/upload");
        xhr.setRequestHeader("X-Comfy-Settings", "1");
        xhr.responseType = "json";
        xhr.upload.onprogress = event => setProgress({ loaded: event.loaded, total: event.lengthComputable ? event.total : 0 });
        xhr.onload = () => {
          if (xhr.status === 201 && xhr.response?.path) resolve();
          else reject(new Error(xhr.response?.error || "파일을 업로드하지 못했습니다. 설정 서버 연결을 확인하세요."));
        };
        xhr.onerror = () => reject(new Error("서버 연결이 끊겼습니다. 목록을 새로고침해 업로드 여부를 확인하세요."));
        xhr.onabort = () => reject(new Error("업로드가 중단되었습니다."));
        xhr.send(body);
      });
      request.current = null;
      notify(`${file.name} 업로드가 완료되었습니다.`);
      await onUploaded();
      onClose();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "파일을 업로드하지 못했습니다.");
    } finally { request.current = null; setBusy(false); }
  }

  return <Modal title={`${label} 업로드`} onClose={onClose} closeDisabled={busy}>
    <form className="form-stack resource-upload-form" onSubmit={event => void upload(event)} aria-busy={busy}>
      <p className="dialog-description">컴퓨터의 {label} 파일을 선택하세요. 기반 모델, 트리거와 미리보기 이미지도 함께 등록할 수 있습니다.</p>
      <label className="field">{label} 파일<input type="file" accept={modelExtensions} required disabled={busy} onChange={event => { setFile(event.target.files?.[0] ?? null); setError(""); }} /></label>
      {file && <p className="field-help resource-upload-file">{file.name} · {bytes(file.size)}</p>}
      <div className="two-columns">
        <label className="field">저장 폴더<select value={folder} disabled={busy || kind === "loras"} onChange={event => setFolder(event.target.value)}>{kind === "loras" ? <option value="loras">LoRA · loras</option> : modelFolders.map(([value, name]) => <option key={value} value={value}>{name}</option>)}</select></label>
        <label className="field">하위 폴더 (선택)<input value={subfolder} maxLength={240} disabled={busy} onChange={event => setSubfolder(event.target.value)} placeholder="예: illustration" /></label>
      </div>
      <label className="field">기반 모델 (선택)<input name="family" maxLength={80} disabled={busy} placeholder="Anima, SDXL 등" /></label>
      <label className="field">트리거 (선택)<textarea name="trigger" rows={2} maxLength={16384} disabled={busy} placeholder="쉼표 또는 줄바꿈으로 구분" /></label>
      <label className="field">미리보기 이미지 (선택)<input ref={previewInput} type="file" accept="image/png,image/jpeg,image/webp" disabled={busy} onChange={event => { setPreview(event.target.files?.[0] ?? null); setError(""); }} /><span className="field-help">PNG, JPEG, WebP · 최대 10 MB</span></label>
      {preview && <div className="resource-upload-preview">{previewUrl && <img src={previewUrl} alt="선택한 미리보기 이미지" />}<button className="text-button" type="button" disabled={busy} onClick={() => { setPreview(null); if (previewInput.current) previewInput.current.value = ""; }}>미리보기 제거</button></div>}
      <p className="field-help">저장 위치: {folder}/{subfolder.trim() && `${subfolder.trim()}/`}{file?.name}<br />같은 이름의 기존 파일은 덮어쓰지 않습니다.</p>
      {busy && <div className="download-job" role="status"><progress aria-label="업로드 진행률" value={progress.total ? progress.loaded : undefined} max={progress.total || 1} /><small>{progress.total && progress.loaded >= progress.total ? "파일 저장 중…" : `업로드 중… ${bytes(progress.loaded)}${progress.total ? ` / ${bytes(progress.total)}` : ""}`}</small></div>}
      {error && <p className="field-error" role="alert">{error}</p>}
      <div className="modal-actions"><button className="button secondary" type="button" disabled={busy} onClick={onClose}>취소</button><button className="button primary" type="submit" disabled={busy || !file}><Upload size={15} />{busy ? "업로드 중…" : "업로드"}</button></div>
    </form>
  </Modal>;
}
