import { useEffect, useRef, useState, type PointerEvent } from "react";
import { Brush, Eraser, LockKeyhole, Move, Play, Redo2, RotateCcw, Sparkles, Undo2, UnlockKeyhole } from "lucide-react";
import { Modal } from "./components";
import { assetRequest, type AssetRecord } from "./asset";
import { AssetEditCanvas, canvasBlob, editMaskColor, loadEditImage, useEditCanvas, type EditCanvasTool } from "./AssetEditCanvas";
import { AssetEditItems, editItemUrl } from "./AssetEditItems";

type EditTarget = { image: AssetRecord; character: string; outfit: string; expression: string; url: string };
type CompositeItem = { id: string; name: string; image: HTMLImageElement; x: number; y: number; width: number; height: number; rotation: number };
type EditOptions = { label: string; defaults: Record<string, number>; edit_tool: string };
const advancedFields = [
  { key: "mask_grow", label: "Mask Grow", min: -512, max: 512, step: 1 },
  { key: "mask_blur", label: "Mask Blur", min: 0, max: 100, step: 0.5 },
  { key: "seed", label: "Seed (-1 랜덤)", min: -1, max: Number.MAX_SAFE_INTEGER, step: 1 },
  { key: "steps", label: "Steps", min: 1, max: 100, step: 1 },
  { key: "cfg", label: "CFG", min: 0, max: 100, step: 0.1 },
  { key: "denoise", label: "Denoise", min: 0, max: 1, step: 0.05 },
] as const;

export function AssetImageEdit({ target, onClose, onQueued }: { target: EditTarget; onClose: () => void; onQueued: () => void }) {
  const [tool, setTool] = useState<EditCanvasTool>("brush");
  const [size, setSize] = useState(48);
  const [opacity, setOpacity] = useState(64);
  const [maskVisible, setMaskVisible] = useState(true);
  const [prompt, setPrompt] = useState("");
  const [english, setEnglish] = useState("");
  const [positive, setPositive] = useState(target.image.positive || "");
  const [negative, setNegative] = useState(target.image.edit_negative_prompt || target.image.negative || "");
  const [options, setOptions] = useState<EditOptions | null>(null);
  const [advanced, setAdvanced] = useState<Record<string, string>>({});
  const [locked, setLocked] = useState(true);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [translating, setTranslating] = useState(false);
  const submitting = useRef(false);
  const translatingRef = useRef(false);
  const [translationStatus, setTranslationStatus] = useState("");
  const [error, setError] = useState("");
  const [revision, setRevision] = useState(0);
  const surface = useEditCanvas(tool, size, busy);
  const [libraryOpen, setLibraryOpen] = useState(false);
  const [items, setItems] = useState<CompositeItem[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const selected = items.find(item => item.id === selectedId);
  const compositeGesture = useRef<{ item: CompositeItem; kind: "move" | "resize" | "rotate"; x: number; y: number } | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    const folder = `asset_mode/characters/${encodeURIComponent(target.character)}/outfits/${encodeURIComponent(target.outfit)}/expressions/${encodeURIComponent(target.expression)}/images`;
    setLoading(true); setError("");
    void Promise.all([assetRequest("asset/edit-options", undefined, controller.signal), assetRequest(folder, undefined, controller.signal)])
      .then(([settings, metadata]) => {
        const latest = metadata.images?.find((image: AssetRecord) => image.filename === target.image.filename);
        if (!latest) throw new Error("선택한 이미지의 최신 정보를 찾지 못했습니다.");
        setOptions(settings as EditOptions);
        setAdvanced(Object.fromEntries(Object.entries(settings.defaults).map(([key, value]) => [key, String(value)])));
        setPositive(latest.positive || ""); setNegative(latest.edit_negative_prompt || latest.negative || "");
      }).catch(cause => { if (!controller.signal.aborted) setError(cause instanceof Error ? cause.message : String(cause)); })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [target.character, target.outfit, target.expression, target.image.filename, revision]);

  function changeMask(action: "clear" | "fill" | "invert") {
    const canvas = surface.canvas.current!;
    const context = canvas.getContext("2d")!;
    if (action === "invert") {
      const data = context.getImageData(0, 0, canvas.width, canvas.height);
      for (let i = 0; i < data.data.length; i += 4) {
        data.data[i] = 22; data.data[i + 1] = 140; data.data[i + 2] = 255; data.data[i + 3] = 255 - data.data[i + 3];
      }
      context.putImageData(data, 0, 0);
    } else {
      context.clearRect(0, 0, canvas.width, canvas.height);
      if (action === "fill") { context.fillStyle = editMaskColor; context.fillRect(0, 0, canvas.width, canvas.height); }
    }
    surface.checkpoint();
  }
  async function translate() {
    if (!prompt.trim() || translatingRef.current || submitting.current) return;
    translatingRef.current = true; setTranslating(true); setTranslationStatus("LLM 큐 대기 및 Edit 프롬프트 작성 중…");
    try {
      const result = await assetRequest("asset_mode/qwen_edit/translate", { text: prompt.trim(), source_prompt: positive });
      if (!result.translated_prompt) throw new Error("작성 결과가 비어 있습니다.");
      setEnglish(result.translated_prompt); setTranslationStatus("작성 완료");
    } catch (cause) { setTranslationStatus(`작성 실패: ${cause instanceof Error ? cause.message : String(cause)}`); }
    finally { translatingRef.current = false; setTranslating(false); }
  }
  async function addItem(record: AssetRecord) {
    const image = await loadEditImage(editItemUrl(record));
    const scale = Math.min(surface.dimensions.width * 0.45 / image.naturalWidth, surface.dimensions.height * 0.45 / image.naturalHeight, 1);
    const item: CompositeItem = { id: crypto.randomUUID(), name: record.name || record.filename, image,
      x: surface.dimensions.width / 2, y: surface.dimensions.height / 2,
      width: image.naturalWidth * scale, height: image.naturalHeight * scale, rotation: 0 };
    setItems(previous => [...previous, item]); setSelectedId(item.id); setTool("composite"); setLibraryOpen(false);
  }
  function updateItem(patch: Partial<CompositeItem>) { setItems(previous => previous.map(item => item.id === selectedId ? { ...item, ...patch } : item)); }
  function removeSelected() { setItems(previous => previous.filter(item => item.id !== selectedId)); setSelectedId(null); }
  function beginComposite(event: PointerEvent<HTMLDivElement>, item: CompositeItem) {
    if (busy || tool !== "composite" || event.button !== 0) return;
    event.preventDefault(); event.stopPropagation();
    event.currentTarget.setPointerCapture(event.pointerId); setSelectedId(item.id);
    const handle = (event.target as HTMLElement).dataset.handle;
    compositeGesture.current = { item, kind: handle === "resize" || handle === "rotate" ? handle : "move", ...surface.point(event) };
  }
  function moveComposite(event: PointerEvent<HTMLDivElement>) {
    const current = compositeGesture.current;
    if (!current) return;
    const point = surface.point(event), { item, kind } = current;
    let patch: Partial<CompositeItem>;
    if (kind === "move") patch = { x: item.x + point.x - current.x, y: item.y + point.y - current.y };
    else if (kind === "rotate") patch = { rotation: item.rotation + (Math.atan2(point.y - item.y, point.x - item.x) - Math.atan2(current.y - item.y, current.x - item.x)) * 180 / Math.PI };
    else {
      const ratio = Math.max(0.05, Math.hypot(point.x - item.x, point.y - item.y) / Math.max(1, Math.hypot(current.x - item.x, current.y - item.y)));
      patch = { width: item.width * ratio, height: item.height * ratio };
    }
    setItems(previous => previous.map(value => value.id === item.id ? { ...value, ...patch } : value));
  }
  function finishComposite(event: PointerEvent<HTMLDivElement>) {
    compositeGesture.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
  }
  async function enqueue() {
    if (submitting.current || translatingRef.current || loading || !options || !surface.dimensions.width) return;
    setError("");
    if (!(english.trim() || prompt.trim())) { setError("편집 지시를 입력하세요."); return; }
    const blocks = negative.match(/\[[^\]\n]{1,40}\]/g);
    if (blocks) { setError(`부정 프롬프트의 [ ] 블록 토큰을 지워주세요: ${blocks.join(", ")}`); return; }
    submitting.current = true; setBusy(true);
    try {
      const canvas = surface.canvas.current!;
      const source = document.createElement("canvas"), mask = document.createElement("canvas");
      for (const output of [source, mask]) { output.width = canvas.width; output.height = canvas.height; }
      const sourceContext = source.getContext("2d")!, maskContext = mask.getContext("2d")!;
      const sourceImage = await loadEditImage(target.url);
      sourceContext.drawImage(sourceImage, 0, 0, source.width, source.height); maskContext.drawImage(canvas, 0, 0);
      // 합성 아이템은 원본에만 얹는다. 마스크까지 넣으면 인페인팅이 그 영역을
      // 다시 생성하면서 아이템을 지워버리므로, 브러시로 칠한 영역만 편집한다.
      for (const item of items) {
        sourceContext.save(); sourceContext.translate(item.x, item.y); sourceContext.rotate(item.rotation * Math.PI / 180);
        sourceContext.drawImage(item.image, -item.width / 2, -item.height / 2, item.width, item.height); sourceContext.restore();
      }
      const pixels = maskContext.getImageData(0, 0, mask.width, mask.height).data;
      if (!pixels.some((value, index) => index % 4 === 3 && value > 0)) throw new Error("편집 마스크를 그리세요.");
      const form = new FormData();
      const [sourceBlob, maskBlob] = await Promise.all([canvasBlob(source), canvasBlob(mask)]);
      form.append("source", sourceBlob, "source.png"); form.append("mask", maskBlob, "mask.png");
      for (const key of ["character", "outfit", "expression"] as const) form.append(key, target[key]);
      form.append("filename", target.image.filename);
      form.append("edit_prompt", english.trim() || prompt.trim()); form.append("edit_prompt_original", prompt.trim() || english.trim());
      form.append("negative_prompt", negative.trim());
      if (!locked) for (const field of advancedFields) form.append(field.key, advanced[field.key]);
      await assetRequest("asset_mode/qwen_edit/enqueue", form); onQueued();
    } catch (cause) { setError(cause instanceof Error ? cause.message : String(cause)); }
    finally { submitting.current = false; setBusy(false); }
  }
  const ready = !busy && !!surface.dimensions.width;
  return <>
    <Modal title="EDIT 툴 호출" className="asset-edit-modal" closeDisabled={busy || translating} onClose={onClose}>
      <p className="asset-edit-subtitle">{options?.label} · {target.character} / {target.outfit} / {target.expression} · {target.image.filename}</p>
      <form className="asset-edit-form" onSubmit={event => { event.preventDefault(); void enqueue(); }}>
        <div className="asset-edit-layout">
          <section className="asset-edit-mask" aria-label="이미지 및 마스크 편집">
            <div className="asset-edit-toolbar" role="group" aria-label="마스크 편집 도구">
              {([["brush", "브러시", Brush], ["eraser", "지우개", Eraser], ["move", "이동", Move]] as const).map(([value, label, Icon]) => <button type="button" className="button" key={value} disabled={!ready} aria-pressed={tool === value} onClick={() => setTool(value)}><Icon size={14} />{label}</button>)}
              <button type="button" className="button" disabled={!ready} onClick={() => setLibraryOpen(true)}>합성 아이템 추가</button>
              <button type="button" className="button" disabled={!ready} aria-pressed={tool === "composite"} onClick={() => setTool("composite")}>합성 아이템 조작</button>
              <button type="button" className="button danger" disabled={!ready || !selected} onClick={removeSelected}>선택 삭제</button>
              <label className="asset-edit-range">크기<input type="range" min={4} max={240} value={size} disabled={busy} onChange={event => setSize(Number(event.target.value))} /><output>{size}</output></label>
              <label className="asset-edit-range">알파<input type="range" min={10} max={100} value={opacity} disabled={busy} onChange={event => setOpacity(Number(event.target.value))} /><output>{opacity}%</output></label>
              <button type="button" className="button" disabled={!ready || !surface.historyState.undo} onClick={() => surface.historyMove(-1)}><Undo2 size={14} />실행취소</button>
              <button type="button" className="button" disabled={!ready || !surface.historyState.redo} onClick={() => surface.historyMove(1)}><Redo2 size={14} />다시실행</button>
              <button type="button" className="button" disabled={!ready} onClick={() => changeMask("clear")}><RotateCcw size={14} />초기화</button>
              <button type="button" className="button" disabled={!ready} onClick={() => changeMask("invert")}>반전</button>
              <button type="button" className="button" disabled={!ready} onClick={() => changeMask("fill")}>전체 선택</button>
              <label className="asset-edit-check"><input type="checkbox" checked={maskVisible} disabled={busy} onChange={event => setMaskVisible(event.target.checked)} />마스크 표시</label>
              <span className="field-help">줌 {Math.round(surface.view.zoom * 100)}%</span>
              <button type="button" className="button" disabled={!ready} onClick={surface.fit}>맞춤</button>
            </div>
            <AssetEditCanvas surface={surface} tool={tool} size={size} busy={busy} imageUrl={target.url} opacity={maskVisible ? opacity / 100 : 0} label="편집할 영역 그리기" onError={setError}>
              {items.map(item => <div key={item.id} className={`asset-edit-composite${selectedId === item.id && tool === "composite" ? " selected" : ""}`} tabIndex={tool === "composite" ? 0 : undefined}
                role="group" aria-label={item.name} style={{ left: item.x, top: item.y, width: item.width, height: item.height, transform: `translate(-50%, -50%) rotate(${item.rotation}deg)`, pointerEvents: tool === "composite" ? "auto" : "none" }}
                onPointerDown={event => beginComposite(event, item)} onPointerMove={moveComposite} onPointerUp={finishComposite} onPointerCancel={finishComposite} onLostPointerCapture={() => { compositeGesture.current = null; }} onFocus={() => setSelectedId(item.id)}
                onKeyDown={event => {
                  if (busy || tool !== "composite") return;
                  if (event.key === "Delete" || event.key === "Backspace") { event.preventDefault(); removeSelected(); }
                  const delta = event.shiftKey ? 10 : 1;
                  if (["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(event.key)) { event.preventDefault(); updateItem({ x: item.x + (event.key === "ArrowLeft" ? -delta : event.key === "ArrowRight" ? delta : 0), y: item.y + (event.key === "ArrowUp" ? -delta : event.key === "ArrowDown" ? delta : 0) }); }
                }}>
                <img src={item.image.src} alt={item.name} draggable={false} />
                {selectedId === item.id && tool === "composite" && <><span className="asset-edit-item-handle resize" data-handle="resize" style={{ width: 14 / surface.displayScale, height: 14 / surface.displayScale }} /><span className="asset-edit-item-handle rotate" data-handle="rotate" style={{ width: 14 / surface.displayScale, height: 14 / surface.displayScale, top: -28 / surface.displayScale }} /></>}
              </div>)}
            </AssetEditCanvas>
            {selected && tool === "composite" && <fieldset className="asset-edit-item-fields" disabled={busy}><legend>{selected.name}</legend>
              {([['x', 'X'], ['y', 'Y'], ['width', '너비'], ['height', '높이'], ['rotation', '회전']] as const).map(([key, label]) => <label className="field" key={key}>{label}<input type="number" step={1} min={key === "width" || key === "height" ? 1 : undefined} value={Math.round(selected[key])} onChange={event => {
                const value = event.currentTarget.valueAsNumber;
                if (!Number.isFinite(value) || ((key === "width" || key === "height") && value < 1)) return;
                if (key === "width" || key === "height") { const ratio = value / selected[key]; updateItem({ width: selected.width * ratio, height: selected.height * ratio }); }
                else updateItem({ [key]: value });
              }} /></label>)}
            </fieldset>}
            <p className="field-help">휠로 확대·축소하고 가운데 버튼을 드래그해 이동하세요.</p>
          </section>
          <section className="asset-edit-fields" aria-label="EDIT 프롬프트 및 설정">
            <label className="field">원본 이미지 긍정 프롬프트<textarea rows={3} value={positive} readOnly /></label>
            <label className="field">편집 지시 원문<textarea rows={4} value={prompt} maxLength={8000} disabled={busy || translating} onChange={event => setPrompt(event.target.value)} placeholder="마스크 영역을 어떻게 바꿀지 입력하세요." /></label>
            <div className="asset-edit-translate"><button type="button" className="button secondary" disabled={busy || translating || loading || !options || !prompt.trim()} onClick={() => void translate()}><Sparkles size={14} />Edit 프롬프트 작성</button><span className="field-help" role="status">{translationStatus || "작성 결과를 검토하거나 영어 지시를 직접 입력하세요."}</span></div>
            <label className="field">Edit Prompt (English)<textarea rows={4} value={english} maxLength={8000} disabled={busy || translating} onChange={event => setEnglish(event.target.value)} placeholder="번역 결과 또는 영어 편집 지시" /></label>
            <label className="field">부정 프롬프트<textarea rows={4} value={negative} maxLength={8000} disabled={busy || loading} onChange={event => setNegative(event.target.value)} /></label>
            <div className="asset-edit-advanced-head"><strong>고급 설정</strong><button type="button" className="button" disabled={busy || loading || !options} aria-pressed={locked} onClick={() => setLocked(value => !value)} title="잠금을 풀어 이번 편집에 사용할 값을 바꿉니다.">{locked ? <LockKeyhole size={14} /> : <UnlockKeyhole size={14} />}{locked ? "잠금" : "잠금 해제"}</button></div>
            <fieldset className="asset-edit-advanced" disabled={busy || locked || loading}>{advancedFields.map(field => <label className="field" key={field.key}>{field.label}<input type="number" min={field.min} max={field.max} step={field.step} required value={advanced[field.key] ?? ""} onChange={event => setAdvanced(previous => ({ ...previous, [field.key]: event.target.value }))} /></label>)}</fieldset>
          </section>
        </div>
        <div className="asset-edit-footer"><div className="asset-edit-status" aria-live="polite">{error ? <p className="settings-feedback error" role="alert">{error}</p> : <p className="field-help">{loading ? "편집 설정과 최신 프롬프트를 불러오는 중…" : busy ? "이미지와 마스크를 등록하는 중…" : `${options?.label || "EDIT"} · 원본 ${surface.dimensions.width} × ${surface.dimensions.height} · 합성 아이템 ${items.length}개`}</p>}{!loading && !options && <button type="button" className="button" onClick={() => setRevision(value => value + 1)}>다시 불러오기</button>}</div><button type="button" className="button" disabled={busy || translating} onClick={onClose}>닫기</button><button type="submit" className="button primary" disabled={!ready || loading || !options || translating || !(prompt.trim() || english.trim())}><Play size={14} />{busy ? "등록 중…" : "GPU 큐에 EDIT 등록"}</button></div>
      </form>
    </Modal>
    {libraryOpen && <AssetEditItems onClose={() => setLibraryOpen(false)} onAdd={addItem} />}
  </>;
}
