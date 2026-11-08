import { useState } from "react";
import { Modal } from "./components";
import { enqueueLoraRefine, imageUrl, type LoraContext, type LoraData } from "./internalLora";

export function BotLoraBatchRefine({ context, detail, onClose, notify }: { context: LoraContext; detail: LoraData; onClose: () => void; notify: (message: string) => void }) {
  const [selected, setSelected] = useState<string[]>([]); const [busy, setBusy] = useState(false); const [error, setError] = useState("");
  const rows = (detail.characters as LoraData[]).flatMap(card => (card.training_images as LoraData[]).map(image => ({
    card, image, key: JSON.stringify([card.unit_key, image.filename]), enabled: !card.skip_training && !!(image.original_positive || image.positive || "").trim(),
  })));
  const enabled = rows.filter(row => row.enabled);
  return <Modal title="일괄 LLM 정제 · 이미지 선택" onClose={onClose} closeDisabled={busy} className="internal-lora-modal internal-lora-picker-modal"><div className="internal-lora-form">
    <p className="field-help">선택한 이미지의 원본 긍정 프롬프트로 정제합니다. 순차 학습 스킵 또는 프롬프트가 없는 이미지는 제외합니다.</p>
    <div className="internal-lora-selection-summary"><label className="internal-lora-check"><input type="checkbox" checked={!!enabled.length && selected.length === enabled.length} disabled={busy || !enabled.length} ref={input => { if (input) input.indeterminate = selected.length > 0 && selected.length < enabled.length; }} onChange={event => setSelected(event.target.checked ? enabled.map(row => row.key) : [])} />전체선택</label><span role="status">{selected.length} / {enabled.length}장 선택</span></div>
    <div className="internal-lora-picker-scroll"><div className="internal-lora-images">{rows.map(({ card, image, key, enabled }) => <label key={key} className={`internal-lora-image internal-lora-image-choice${selected.includes(key) ? " selected" : ""}`}>
      <img src={imageUrl({ ...context, character: card }, image)} alt={`${card.name} · ${card.visual_card_label}`} loading="lazy" /><span className="internal-lora-image-caption"><strong>{card.name} · {card.visual_card_label}</strong><span className="internal-lora-check"><input type="checkbox" checked={selected.includes(key)} disabled={busy || !enabled} onChange={event => setSelected(current => event.target.checked ? [...current, key] : current.filter(value => value !== key))} /><span>{image.filename}</span></span>{!enabled && <small>{card.skip_training ? "순차 학습 스킵" : "프롬프트 없음"}</small>}</span>
    </label>)}</div></div>
    {error && <p className="field-error" role="alert">{error}</p>}
    <div className="internal-lora-picker-footer"><span>{selected.length}장 선택</span><div className="internal-lora-actions"><button className="button" disabled={busy} onClick={onClose}>취소</button><button className="button primary" disabled={busy || !selected.length} onClick={async () => {
      setBusy(true); setError("");
      try { for (const row of rows.filter(row => row.enabled && selected.includes(row.key))) { await enqueueLoraRefine({ ...context, character: row.card }, [row.image]); setSelected(current => current.filter(key => key !== row.key)); } notify("선택한 이미지를 LLM 정제 큐에 추가했습니다."); onClose(); }
      catch (cause) { setError((cause as Error).message); } finally { setBusy(false); }
    }}>{busy ? "큐에 추가 중…" : "정제 시작"}</button></div></div>
  </div></Modal>;
}
