import { useState, type FormEvent } from "react";
import { FolderInput } from "lucide-react";
import { Modal } from "./components";
import { modelFolders } from "./data";
import { resourceDisplayName, type MovedResource, type ResourceEntry, type Resources } from "./useResources";

export function ResourceMove({ item, resources, onClose, onMoved }: {
  item: ResourceEntry; resources: Resources; onClose: () => void;
  onMoved: (item: MovedResource, warning: string) => void;
}) {
  const isLora = "category" in item;
  const parts = (item.path || "").split("/");
  const filename = parts.at(-1) || item.name;
  const currentFolder = parts.length > 1 ? parts[0] : "";
  const [folder, setFolder] = useState(isLora ? "loras" : currentFolder === "checkpoints" && /anima/i.test(item.family) ? "diffusion_models" : currentFolder || "diffusion_models");
  const [subfolder, setSubfolder] = useState(parts.slice(1, -1).join("/"));
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const folders = isLora ? [["loras", "LoRA"]] : [...modelFolders, ["unet", "Diffusion (UNet)"], ["clip", "Text encoder (CLIP)"], ["vae_approx", "VAE approx"]];
  const target = `${folder}/${subfolder.trim() ? `${subfolder.trim()}/` : ""}${filename}`;
  const blockedReason = [...resources.models, ...resources.loras].find(value => value.id === item.id)?.moveBlockedReason ?? item.moveBlockedReason;

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (busy || blockedReason || target === item.path) return;
    setBusy(true);
    setError("");
    try {
      const result = await resources.move(item.id, folder, subfolder.trim());
      onMoved(result.item, result.warning);
      onClose();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "파일을 이동하지 못했습니다.");
    } finally { setBusy(false); }
  }

  return <Modal title={`${isLora ? "LoRA" : "모델"} 이동`} onClose={onClose} closeDisabled={busy}>
    <form className="form-stack" onSubmit={event => void submit(event)} aria-busy={busy}>
      <p><strong>{resourceDisplayName(item)}</strong> 파일을 옮길 폴더를 선택하세요.</p>
      <div className="field"><span>현재 위치</span><p className="resource-file-path">{item.path}</p></div>
      <label className="field">대상 폴더<select value={folder} disabled={busy || isLora} onChange={event => setFolder(event.target.value)}>
        {!folders.some(([value]) => value === folder) && <option value={folder} disabled>{folder}</option>}
        {folders.map(([value, label]) => <option key={value} value={value}>{label} · {value}</option>)}
      </select></label>
      <label className="field">하위 폴더 (선택)<input value={subfolder} maxLength={240} disabled={busy} onChange={event => setSubfolder(event.target.value)} placeholder="예: illustration" /></label>
      {!isLora && <p className="field-help">Anima 모델은 Diffusion · diffusion_models 폴더를 선택하세요.</p>}
      <div className="field"><span>이동 후 위치</span><p className="resource-file-path">{target}</p></div>
      <p className="field-help">썸네일과 메타데이터도 함께 옮깁니다. 다른 모델과 공유하는 파일은 원래 폴더에도 남겨둡니다. 같은 이름의 파일은 덮어쓰지 않습니다.</p>
      <p className="field-help">이동 후 워크플로우의 모델 선택을 확인하세요. 하위 폴더를 바꾸면 모델 경로도 다시 선택해야 합니다.</p>
      {(error || blockedReason) && <p className="field-error" role="alert">{error || blockedReason}</p>}
      <div className="modal-actions"><button className="button secondary" type="button" disabled={busy} onClick={onClose}>취소</button><button className="button primary" type="submit" disabled={busy || !!blockedReason || target === item.path}><FolderInput size={15} />{busy ? "이동 중…" : "이동"}</button></div>
    </form>
  </Modal>;
}
