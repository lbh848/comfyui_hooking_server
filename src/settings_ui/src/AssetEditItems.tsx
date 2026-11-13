import { useEffect, useRef, useState, type PointerEvent } from "react";
import { Crop, Eraser, Move, RotateCcw, Undo2, Upload, ZoomIn, ZoomOut } from "lucide-react";
import { Modal } from "./components";
import { assetRequest, type AssetRecord } from "./asset";
import { AssetEditCanvas, canvasBlob, loadEditImage, useEditCanvas, type EditPoint } from "./AssetEditCanvas";

const itemPath = "asset_mode/qwen_edit/composite_items";
export const editItemUrl = (item: AssetRecord) => `/api/asset/${itemPath}/${encodeURIComponent(item.filename)}?v=${item.modified_at || 0}`;

export function AssetEditItems({ onClose, onAdd }: { onClose: () => void; onAdd: (record: AssetRecord) => Promise<void> }) {
  const [tool, setTool] = useState<"move" | "eraser" | "crop">("eraser");
  const [size, setSize] = useState(48);
  const [busy, setBusy] = useState(false);
  const working = useRef(false);
  const surface = useEditCanvas(tool, size, busy);
  const original = useRef<ImageData | null>(null);
  const upload = useRef<HTMLInputElement>(null);
  const [name, setName] = useState("");
  const [items, setItems] = useState<AssetRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [status, setStatus] = useState("편집할 이미지를 선택하세요.");
  const [deleting, setDeleting] = useState<AssetRecord | null>(null);
  const [crop, setCrop] = useState<{ x: number; y: number; width: number; height: number } | null>(null);
  const cropStart = useRef<EditPoint | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    void assetRequest(itemPath, undefined, controller.signal).then(result => setItems(result.items || []))
      .catch(cause => { if (!controller.signal.aborted) setError(String(cause)); }).finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, []);
  async function refresh() { setItems((await assetRequest(itemPath)).items || []); }
  async function action(task: () => Promise<void>) {
    if (working.current) return;
    working.current = true; setBusy(true); setError("");
    try { await task(); } catch (cause) { setError(cause instanceof Error ? cause.message : String(cause)); }
    finally { working.current = false; setBusy(false); }
  }
  async function loadFile(file: File) {
    await action(async () => {
      if (!file.type.startsWith("image/")) throw new Error("이미지 파일을 선택하세요.");
      const url = URL.createObjectURL(file);
      try {
        const image = await loadEditImage(url);
        surface.initialize(image.naturalWidth, image.naturalHeight, image);
        original.current = surface.canvas.current!.getContext("2d")!.getImageData(0, 0, image.naturalWidth, image.naturalHeight);
        setName(file.name.replace(/\.[^.]+$/, "").slice(0, 80)); setCrop(null);
        setStatus(`${image.naturalWidth} × ${image.naturalHeight} · 지우개나 자르기로 편집하세요.`);
      } finally { URL.revokeObjectURL(url); }
    });
  }
  function cropPoint(event: PointerEvent<HTMLDivElement>) {
    const point = surface.point(event);
    return { x: Math.max(0, Math.min(surface.dimensions.width, point.x)), y: Math.max(0, Math.min(surface.dimensions.height, point.y)) };
  }
  function applyCrop() {
    if (!crop || crop.width < 1 || crop.height < 1) return;
    const context = surface.canvas.current!.getContext("2d")!;
    const snapshot = context.getImageData(Math.round(crop.x), Math.round(crop.y), Math.max(1, Math.floor(crop.width)), Math.max(1, Math.floor(crop.height)));
    surface.restore(snapshot); surface.checkpoint(); surface.fit(); setCrop(null);
  }
  async function saveAndAdd() {
    await action(async () => {
      const source = surface.canvas.current!;
      const pixels = source.getContext("2d")!.getImageData(0, 0, source.width, source.height).data;
      let left = source.width, top = source.height, right = -1, bottom = -1;
      for (let y = 0; y < source.height; y++) for (let x = 0; x < source.width; x++) if (pixels[(y * source.width + x) * 4 + 3]) {
        left = Math.min(left, x); right = Math.max(right, x); top = Math.min(top, y); bottom = Math.max(bottom, y);
      }
      if (right < left) throw new Error("이미지가 비어 있습니다. 원본을 복원하거나 새 이미지를 선택하세요.");
      const trimmed = document.createElement("canvas"); trimmed.width = right - left + 1; trimmed.height = bottom - top + 1;
      trimmed.getContext("2d")!.drawImage(source, left, top, trimmed.width, trimmed.height, 0, 0, trimmed.width, trimmed.height);
      const form = new FormData(); form.append("image", await canvasBlob(trimmed), "item.png"); form.append("name", name.trim());
      const result = await assetRequest(itemPath, form);
      await refresh();
      await onAdd(result.item);
    });
  }
  const loaded = !!surface.dimensions.width;
  return <>
    <Modal title="합성 아이템 편집 · 전역 아이템 박스" className="asset-edit-modal asset-edit-items-modal" backgroundable={false} closeDisabled={busy} onClose={onClose}>
      <div className="asset-edit-items-layout">
        <section className="asset-edit-mask" onDragOver={event => event.preventDefault()} onDrop={event => { event.preventDefault(); const file = event.dataTransfer.files[0]; if (file && !busy) void loadFile(file); }}>
          <div className="asset-edit-toolbar" role="group" aria-label="합성 아이템 편집 도구">
            <input ref={upload} type="file" accept="image/*" hidden onChange={event => { const file = event.target.files?.[0]; event.target.value = ""; if (file) void loadFile(file); }} />
            <button type="button" className="button" disabled={busy} onClick={() => upload.current?.click()}><Upload size={14} />새 이미지 업로드</button>
            {([["move", "이동", Move], ["eraser", "지우개", Eraser], ["crop", "자르기", Crop]] as const).map(([value, label, Icon]) => <button type="button" className="button" key={value} disabled={busy || !loaded} aria-pressed={tool === value} onClick={() => setTool(value)}><Icon size={14} />{label}</button>)}
            <button type="button" className="button" disabled={busy || !crop?.width || !crop?.height} onClick={applyCrop}>자르기 적용</button>
            <button type="button" className="button" aria-label="축소" disabled={busy || !loaded} onClick={() => surface.zoomAt(surface.view.zoom / 1.25)}><ZoomOut size={14} /></button>
            <button type="button" className="button" aria-label="확대" disabled={busy || !loaded} onClick={() => surface.zoomAt(surface.view.zoom * 1.25)}><ZoomIn size={14} /></button>
            <button type="button" className="button" disabled={busy || !loaded} onClick={surface.fit}>맞춤</button>
            <label className="asset-edit-range">지우개<input type="range" min={4} max={240} value={size} disabled={busy} onChange={event => setSize(Number(event.target.value))} /><output>{size}</output></label>
            <button type="button" className="button" disabled={busy || !surface.historyState.undo} onClick={() => { surface.historyMove(-1); setCrop(null); }}><Undo2 size={14} />실행취소</button>
            <button type="button" className="button" disabled={busy || !original.current} onClick={() => { surface.restore(original.current!); surface.checkpoint(); surface.fit(); setCrop(null); }}><RotateCcw size={14} />원본 복원</button>
          </div>
          <AssetEditCanvas surface={surface} tool={tool} size={size} busy={busy} label="합성 아이템 편집 이미지">
            {tool === "crop" && <div className="asset-edit-crop-layer" onPointerDown={event => {
              if (busy || event.button !== 0) return;
              event.stopPropagation(); event.currentTarget.setPointerCapture(event.pointerId); cropStart.current = cropPoint(event); setCrop(null);
            }} onPointerMove={event => {
              if (!cropStart.current) return;
              const point = cropPoint(event), start = cropStart.current;
              setCrop({ x: Math.min(start.x, point.x), y: Math.min(start.y, point.y), width: Math.abs(point.x - start.x), height: Math.abs(point.y - start.y) });
            }} onPointerUp={event => { cropStart.current = null; if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId); }} onLostPointerCapture={() => { cropStart.current = null; }}>
              {crop && <span className="asset-edit-crop-box" style={{ left: crop.x, top: crop.y, width: crop.width, height: crop.height }} />}
            </div>}
          </AssetEditCanvas>
          <p className="field-help" role="status">{status}</p>
        </section>
        <aside className="asset-edit-library">
          <div className="asset-edit-library-header"><strong>서버 전역 아이템 박스</strong><button type="button" className="button" disabled={busy} onClick={() => void action(refresh)}>새로고침</button></div>
          {loading && <p role="status">아이템 목록을 불러오는 중…</p>}
          {!loading && !items.length && <p className="field-help">저장된 아이템이 없습니다. 새 이미지를 업로드하세요.</p>}
          <div className="asset-edit-library-grid">{items.map(item => <article key={item.filename}>
            <img src={editItemUrl(item)} alt={item.name || item.filename} loading="lazy" />
            <strong>{item.name || item.filename}</strong><small>{item.width} × {item.height}</small>
            <div className="asset-actions"><button type="button" className="button" disabled={busy} onClick={() => void action(() => onAdd(item))}>이미지에 추가</button><button type="button" className="button danger" disabled={busy} onClick={() => setDeleting(item)}>삭제</button></div>
          </article>)}</div>
        </aside>
      </div>
      {error && <p className="settings-feedback error" role="alert">{error}</p>}
      <div className="asset-edit-footer"><label className="field">아이템 이름<input value={name} maxLength={80} disabled={busy} placeholder="예: 검은 리본" onChange={event => setName(event.target.value)} /></label><button type="button" className="button" disabled={busy} onClick={onClose}>닫기</button><button type="button" className="button primary" disabled={busy || !loaded} onClick={() => void saveAndAdd()}>{busy ? "처리 중…" : "파일로 저장하고 추가"}</button></div>
    </Modal>
    {deleting && <Modal title="합성 아이템 삭제" backgroundable={false} closeDisabled={busy} onClose={() => setDeleting(null)}><p>{deleting.name || deleting.filename} 아이템을 삭제할까요?</p><div className="modal-actions"><button type="button" className="button" disabled={busy} onClick={() => setDeleting(null)}>취소</button><button type="button" className="button danger" disabled={busy} onClick={() => void action(async () => { await assetRequest(`${itemPath}/${encodeURIComponent(deleting.filename)}`, undefined, undefined, "DELETE"); setDeleting(null); await refresh(); })}>삭제</button></div>{error && <p role="alert">{error}</p>}</Modal>}
  </>;
}
